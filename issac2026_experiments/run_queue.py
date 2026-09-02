#!/usr/bin/env python3
"""Run queue for the Table 8 campaign: 26 configurations x 3 seeds (42, 123, 7).

Environment: GPUS (comma-separated, default "0,1"), GPUS_IF_IDLE, CALT_SRC,
SAGE_PYTHON, SAGE_BIN, EXP_ROOT.

Upgrades over v1:
- RESUME: on start, any job whose log already ends with "Success rate:" is
  marked done and skipped (safe across reboots).
- PEAK GPU MEMORY: a sampler thread polls nvidia-smi per run PID every 15 s;
  the peak (MiB) is logged at DONE and written to <results_dir>/peak_gpu_mem_mib.txt
  so every table row carries its own memory figure.
- MULTI-SEED: full seed-42 pass first, then seeds 123 and 7. Seed variants are
  lazy config copies with seed/save_dir/wandb suffix rewritten.
- GPU slots from GPUS; those in GPUS_IF_IDLE are used only when fully idle (<500 MiB).
"""
import os, re, subprocess, threading, time, traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(os.environ.get("EXP_ROOT", Path(__file__).resolve().parent))
STATUS = ROOT / "QUEUE_STATUS.txt"
LOGF = ROOT / "queue2.log"

PY = os.environ.get("SAGE_PYTHON", "python3")
ENV = dict(os.environ)
ENV["PATH"] = os.environ.get("SAGE_BIN", "") + os.pathsep + ENV.get("PATH", "")
ENV["PYTHONPATH"] = os.environ.get("CALT_SRC", "")
ENV["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

OUR_GPUS = [int(g) for g in os.environ.get("GPUS", "0,1").split(",")]
IDLE_EXTRA = [int(g) for g in os.environ.get("GPUS_IF_IDLE", "").split(",") if g]
MAX_RETRIES = 2
POLL = 120
SEEDS = [42, 123, 7]

GB = ROOT / "groebner"
PM = ROOT / "polynomial_multiplication"
PR = ROOT / "polynomial_reduction"


def seed_cfg(cwd: Path, cfg_rel: str, seed: int) -> str:
    """Return the config path for this seed, creating a variant lazily for != 42."""
    if seed == 42:
        return cfg_rel
    src = cwd / cfg_rel
    dst_rel = cfg_rel.replace(".yaml", f"_seed{seed}.yaml")
    dst = cwd / dst_rel
    if not dst.exists():
        t = src.read_text()
        t = re.sub(r"seed:\s*\d+", f"seed: {seed}", t)
        t = re.sub(r"(save_dir:\s*\S+)", rf"\1_seed{seed}", t)
        dst.write_text(t)
    return dst_rel


def J(name, cwd, cfg_flag, cfg_rel, extra, log_dir, requires, seed):
    return {"name": f"{name}_s{seed}", "cwd": str(cwd), "cfg_flag": cfg_flag,
            "cfg_rel": cfg_rel, "extra": extra,
            "log": str(cwd / f"{log_dir}{'' if seed == 42 else f'_seed{seed}'}" / "train.log"),
            "requires": str(cwd / requires), "seed": seed, "retries": 0}


QUEUE = []
for seed in SEEDS:  # full pass per seed: 42 first (fills Table 8), then 123, 7
    for field in ("GF7", "ZZ", "GF31", "GF97"):
        for which in ("standard", "monomial"):
            for mode in ("last_element", "full"):
                QUEUE.append(J(f"pm_{field}_{which}_{mode}", PM,
                               "--train_config_path", f"configs/{field}/train_repr_{which}.yaml",
                               ["--data_config_path", f"configs/{field}/data.yaml",
                                "--target_mode", mode,
                                "--wandb_runname_postfix", f"{which}_{mode}_s{seed}"],
                               f"results_{field}_repr_{which}_{mode}",
                               f"data/{field}/test_stats.yaml", seed))
    for field in ("GF7", "ZZ", "GF31", "GF97"):
        for which in ("standard", "monomial"):
            QUEUE.append(J(f"pr_{field}_{which}", PR,
                           "--train_config_path", f"configs/{field}/train_repr_{which}.yaml",
                           ["--data_config_path", f"configs/{field}/data.yaml",
                            "--target_mode", "full",
                            "--wandb_runname_postfix", f"{which}_s{seed}"],
                           f"results_{field}_repr_{which}",
                           f"data/{field}/test_stats.yaml", seed))
    for which in ("standard", "monomial"):
        QUEUE.append(J(f"gb_repr_{which}", GB,
                       "--config_path", f"configs/train_GF7_repr_{which}.yaml",
                       ["--data_config_path", "configs/data_GF7.yaml",
                        "--training_order", "degrevlex", "--expanded_form"],
                       f"results_GF7_repr_{which}",
                       "data/GF7/test_stats.yaml", seed))

running = {}
done = []
peaks_lock = threading.Lock()


def log(msg):
    line = f"[{datetime.now():%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(LOGF, "a") as f:
        f.write(line + "\n")


def already_done(j):
    p = Path(j["log"])
    return p.exists() and "Success rate:" in p.read_text()[-2000:]


def gpu_mem():
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used",
                                       "--format=csv,noheader,nounits"], text=True, timeout=30)
        return {int(l.split(",")[0]): int(l.split(",")[1]) for l in out.strip().splitlines()}
    except Exception:
        return {}


def proc_gpu_mem():
    """pid -> MiB from nvidia-smi compute-apps."""
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                                       "--format=csv,noheader,nounits"], text=True, timeout=30)
        d = {}
        for l in out.strip().splitlines():
            pid, mem = l.split(",")
            d[int(pid)] = int(mem)
        return d
    except Exception:
        return {}


def mem_sampler():
    while True:
        try:
            pm = proc_gpu_mem()
            with peaks_lock:
                for info in running.values():
                    mib = pm.get(info["proc"].pid, 0)
                    if mib > info.get("peak", 0):
                        info["peak"] = mib
        except Exception:
            pass
        time.sleep(15)


def free_slots():
    mem = gpu_mem()
    slots = [g for g in OUR_GPUS if g not in running]
    slots += [g for g in IDLE_EXTRA if g not in running and mem.get(g, 99999) < 500]
    return slots


def next_ready_job():
    for i, j in enumerate(QUEUE):
        if Path(j["requires"]).exists():
            return QUEUE.pop(i)
    return None


def launch(j, gpu):
    cfg_rel = seed_cfg(Path(j["cwd"]), j["cfg_rel"], j["seed"])
    Path(j["log"]).parent.mkdir(parents=True, exist_ok=True)
    env = dict(ENV); env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    fh = open(j["log"], "a")
    args = [PY, "-u", "train.py", j["cfg_flag"], cfg_rel] + j["extra"]
    p = subprocess.Popen(args, cwd=j["cwd"], env=env, stdout=fh, stderr=subprocess.STDOUT)
    running[gpu] = {"proc": p, "job": j, "fh": fh, "start": time.time(), "peak": 0}
    log(f"LAUNCH {j['name']} GPU{gpu} pid={p.pid} try={j['retries']}")


def reduce_test_batch(j):
    cfgp = Path(j["cwd"]) / seed_cfg(Path(j["cwd"]), j["cfg_rel"], j["seed"])
    try:
        t = cfgp.read_text()
        m = re.search(r"test_batch_size:\s*(\d+)", t)
        if m:
            nb = max(2, int(m.group(1)) // 2)
            cfgp.write_text(re.sub(r"test_batch_size:\s*\d+", f"test_batch_size: {nb}", t))
            log(f"  {j['name']}: test_batch_size -> {nb}")
    except Exception as e:
        log(f"  reduce_batch err {e}")


def reap():
    for gpu, info in list(running.items()):
        p = info["proc"]
        if p.poll() is None:
            continue
        j = info["job"]
        info["fh"].close()
        with peaks_lock:
            peak = info.get("peak", 0)
        tail = ""
        try:
            tail = Path(j["log"]).read_text()[-3000:]
        except Exception:
            pass
        ok = p.returncode == 0 and "Success rate" in tail
        del running[gpu]
        if ok:
            m = re.findall(r"Success rate: ([0-9.]+)%", tail)
            sr = m[-1] if m else "?"
            try:
                (Path(j["log"]).parent / "peak_gpu_mem_mib.txt").write_text(f"{peak}\n")
            except Exception:
                pass
            log(f"DONE {j['name']} success={sr}% peak_mem={peak}MiB")
            done.append((j["name"], f"{sr}% peak={peak}MiB"))
        else:
            if j["retries"] < MAX_RETRIES:
                j["retries"] += 1
                if "out of memory" in tail.lower():
                    reduce_test_batch(j)
                log(f"RETRY {j['name']} rc={p.returncode} (try {j['retries']})")
                QUEUE.insert(0, j)
            else:
                log(f"FAILED {j['name']} rc={p.returncode}")
                done.append((j["name"], f"FAILED rc={p.returncode}"))


def write_status():
    with open(STATUS, "w") as f:
        f.write(f"=== QUEUE2 STATUS @ {datetime.now():%m-%d %H:%M:%S} ===\n\nRUNNING:\n")
        for g, i in running.items():
            f.write(f"  GPU{g}: {i['job']['name']} ({int(time.time()-i['start'])//60} min, peak {i.get('peak',0)}MiB)\n")
        f.write(f"\nQUEUED ({len(QUEUE)}):\n")
        for j in QUEUE[:15]:
            f.write(f"  {j['name']}\n")
        if len(QUEUE) > 15:
            f.write(f"  ... +{len(QUEUE)-15}\n")
        f.write(f"\nFINISHED ({len(done)}):\n")
        for n, s in done:
            f.write(f"  {n}: {s}\n")


def main():
    skipped = 0
    for j in list(QUEUE):
        if already_done(j):
            m = re.findall(r"Success rate: ([0-9.]+)%", Path(j["log"]).read_text()[-2000:])
            done.append((j["name"], f"{m[-1] if m else '?'}% (pre-existing)"))
            QUEUE.remove(j)
            skipped += 1
    log(f"=== queue2 start: {len(QUEUE)} jobs ({skipped} already done), seeds {SEEDS} ===")
    threading.Thread(target=mem_sampler, daemon=True).start()
    while True:
        try:
            reap()
            for gpu in free_slots():
                j = next_ready_job()
                if j is None:
                    break
                launch(j, gpu)
            write_status()
            if not QUEUE and not running:
                log("=== ALL JOBS DONE ===")
                write_status()
                return
        except Exception:
            log("LOOP ERR:\n" + traceback.format_exc())
        time.sleep(POLL)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("FATAL:\n" + traceback.format_exc())

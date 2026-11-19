"""Measure the size of a Groebner basis, lex vs degrevlex: degree and terms.

Companion to ``measure_gb_timing.py``.  That script answers "how long does the
computer algebra system take"; this one answers "how much does the model have to
generate", which is the quantity Table 7 of the paper reports.

The usual explanation for the learning gap between the two orders is that
lexicographic bases blow up in degree *and* in the number of terms, so the
autoregressive model has a longer, more error-prone sequence to produce.  The
degree part and the length part are separable, and at the paper's problem size
they do not point the same way, so both are measured here rather than assumed.

The systems are drawn exactly as ``measure_gb_timing.py`` draws them — same
sampler config, same seed sequence, same rejection rule over QQ — so the two
tables describe one sample set.  The basis is the one the dataset generator
stores as the target, i.e. ``ideal.groebner_basis()``, not a separately reduced
copy: the point is to describe what the model is asked to generate.

Reported per order (see ``--output`` for the full record):

  mean_degree_per_poly     total degree of a basis element, averaged over every
                           element of every system  <- Table 7, "Mean degree"
  mean_terms_per_system    number of monomials in the whole basis, averaged over
                           systems                  <- Table 7, "Mean # terms"
  mean_max_degree          the largest degree in a basis, averaged over systems
  mean_terms_per_poly      monomials per basis element
  mean_basis_size          number of elements in the basis

Usage (from issac2026_experiments/groebner):

    sage -python measure_gb_basis_stats.py --config_path configs/data_GF7.yaml
    sage -python measure_gb_basis_stats.py --config_path configs/data.yaml

Results are appended to ``gb_basis_stats.json`` next to this file.
"""

import json
import os

import click

# Initialize Sage's polynomial ring stack first, as generate_dataset.py does.
import sage.all  # noqa: F401
import sage.misc.randstate as randstate  # type: ignore
from cysignals.alarm import AlarmInterrupt, alarm, cancel_alarm  # type: ignore
from omegaconf import OmegaConf
from sage.all import GF, QQ, ZZ, PolynomialRing  # type: ignore

from calt.dataset.sagemath.utils.polynomial_sampler import PolynomialSampler


def _has_large_rational_coefficients(polynomials, threshold: int = 100) -> bool:
    """Same acceptance test as generate_dataset.py, so the systems match."""
    for p in polynomials:
        for c in p.coefficients():
            try:
                num = c.numerator()
                den = c.denominator()
            except Exception:
                continue
            if abs(int(num)) >= threshold or abs(int(den)) >= threshold:
                return True
    return False


def _base_ring(field_str: str):
    if field_str == "QQ":
        return QQ
    if field_str == "ZZ":
        return ZZ
    if field_str.startswith("GF"):
        return GF(int(field_str[2:]))
    raise ValueError(f"unsupported field_str: {field_str}")


def _basis_stats(ring, F) -> dict:
    """Degree and term count of the basis of ``F`` computed in ``ring``."""
    G = list(ring.ideal([ring(f) for f in F]).groebner_basis())
    degrees = [int(g.total_degree()) for g in G]
    terms = [len(g.monomials()) for g in G]
    return {
        "size": len(G),
        "degrees": degrees,
        "terms": terms,
        "max_degree": max(degrees) if degrees else 0,
        "total_terms": sum(terms),
    }


def _summarize(records: list[dict]) -> dict:
    """Aggregate the per-system records of one order."""
    n = len(records)
    all_degrees = [d for r in records for d in r["degrees"]]
    all_terms = [t for r in records for t in r["terms"]]
    n_polys = len(all_degrees)
    return {
        "n_systems": n,
        "n_polynomials": n_polys,
        # Table 7 rows.
        "mean_degree_per_poly": sum(all_degrees) / n_polys if n_polys else 0.0,
        "mean_terms_per_system": sum(r["total_terms"] for r in records) / n,
        # Context for the two rows above.
        "mean_max_degree": sum(r["max_degree"] for r in records) / n,
        "mean_terms_per_poly": sum(all_terms) / n_polys if n_polys else 0.0,
        "mean_basis_size": sum(r["size"] for r in records) / n,
    }


@click.command()
@click.option(
    "--config_path",
    type=click.Path(exists=True),
    default="configs/data_GF7.yaml",
    help="Data config of the dataset to describe (same sampler settings).",
)
@click.option(
    "--num_samples",
    type=int,
    default=1000,
    help="Number of systems F=[f1,f2] to measure.",
)
@click.option(
    "--root_seed",
    type=int,
    default=None,
    help="Seed offset; defaults to the dataset's root_seed. Keep it equal to "
    "the one measure_gb_timing.py used to describe the same sample set.",
)
@click.option(
    "--output",
    type=click.Path(),
    default="gb_basis_stats.json",
    help="Where to append the measurement.",
)
@click.option(
    "--symbols",
    type=str,
    default=None,
    help="Override the sampler's variables, e.g. 'x,y,z'.",
)
@click.option(
    "--max_degree",
    type=int,
    default=None,
    help="Override the sampler's max_degree.",
)
@click.option(
    "--num_polynomials",
    "num_polynomials_opt",
    type=int,
    default=None,
    help="Override the number of polynomials in F.",
)
@click.option(
    "--timeout_s",
    type=float,
    default=60.0,
    help="Skip (and count) systems whose basis exceeds this budget.",
)
def main(
    config_path: str,
    num_samples: int,
    root_seed: int,
    output: str,
    symbols: str,
    max_degree: int,
    num_polynomials_opt: int,
    timeout_s: float,
) -> None:
    cfg = OmegaConf.load(config_path)
    sampler_cfg = dict(OmegaConf.to_container(cfg.sampler, resolve=True))
    if symbols is not None:
        sampler_cfg["symbols"] = symbols
    if max_degree is not None:
        sampler_cfg["max_degree"] = max_degree
    field_str = sampler_cfg["field_str"]
    symbols = sampler_cfg["symbols"]
    num_polynomials = num_polynomials_opt or int(
        cfg.problem_generator.get("num_polynomials", 2)
    )
    if root_seed is None:
        root_seed = int(cfg.dataset.get("root_seed", 42))

    sampler = PolynomialSampler(**sampler_cfg)
    base = _base_ring(field_str)
    names = [s.strip() for s in symbols.split(",")]
    R_degrevlex = PolynomialRing(base, names, order="degrevlex")
    R_lex = PolynomialRing(base, names, order="lex")
    is_QQ = base == QQ

    degrevlex_records: list[dict] = []
    lex_records: list[dict] = []
    seed = root_seed
    accepted = 0
    timed_out = 0
    while accepted < num_samples:
        randstate.set_random_seed(seed)
        seed += 1
        F = sampler.sample(num_samples=num_polynomials)

        # QQ rejects systems with 3-digit numerators/denominators, as the
        # dataset generator does; the check needs a basis, computed here.
        if is_QQ:
            G_check = list(
                R_degrevlex.ideal([R_degrevlex(f) for f in F]).groebner_basis()
            )
            if _has_large_rational_coefficients(list(F) + G_check, threshold=100):
                continue

        try:
            alarm(timeout_s)
            dr = _basis_stats(R_degrevlex, F)
            lex = _basis_stats(R_lex, F)
            cancel_alarm()
        except AlarmInterrupt:
            cancel_alarm()
            timed_out += 1
            continue

        degrevlex_records.append(dr)
        lex_records.append(lex)
        accepted += 1
        if accepted % 100 == 0:
            print(f"{accepted}/{num_samples} systems measured", flush=True)

    dr_stats = _summarize(degrevlex_records)
    lex_stats = _summarize(lex_records)

    record = {
        "config_path": config_path,
        "field": field_str,
        "symbols": symbols,
        "num_polynomials": num_polynomials,
        "max_degree": sampler_cfg["max_degree"],
        "num_variables": len(names),
        "num_samples": len(degrevlex_records),
        "timed_out": timed_out,
        "timeout_s": timeout_s,
        "root_seed": root_seed,
        "degrevlex": dr_stats,
        "lex": lex_stats,
        # The two ratios the Section 5.4 discussion turns on: lex bases are of
        # higher degree, but that does not by itself make them longer.
        "ratio_degree_per_poly": lex_stats["mean_degree_per_poly"]
        / dr_stats["mean_degree_per_poly"],
        "ratio_terms_per_system": lex_stats["mean_terms_per_system"]
        / dr_stats["mean_terms_per_system"],
    }

    print(json.dumps(record, indent=2))
    print(
        "\nTable 7 rows (lex vs degrevlex):\n"
        f"  Mean degree    {lex_stats['mean_degree_per_poly']:.2f}"
        f"  {dr_stats['mean_degree_per_poly']:.2f}\n"
        f"  Mean # terms   {lex_stats['mean_terms_per_system']:.2f}"
        f"  {dr_stats['mean_terms_per_system']:.2f}"
    )

    existing = []
    if os.path.exists(output):
        with open(output) as fh:
            existing = json.load(fh)
    existing.append(record)
    with open(output, "w") as fh:
        json.dump(existing, fh, indent=2)
    print(f"appended to {output}")


if __name__ == "__main__":
    main()

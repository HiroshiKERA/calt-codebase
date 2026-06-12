"""Generate dataset for [TASK NAME] toy experiment."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

# TODO: replace 'task_template' with your task module name
from task_template.core.generator import generate_split
from shared.paths import config_dir, data_dir
from shared.seed import set_seed
from omegaconf import OmegaConf

if __name__ == "__main__":
    cfg = OmegaConf.load(config_dir(__file__) / "data.yaml")
    d = cfg.dataset
    set_seed(d.root_seed)

    print(f"Generating {d.num_train_samples} train samples...")
    train = generate_split(d.num_train_samples, tag="train", root_seed=d.root_seed)

    print(f"Generating {d.num_test_samples} test samples...")
    test = generate_split(d.num_test_samples, tag="test", root_seed=d.root_seed)

    # TODO: call save_dataset or DatasetPipeline.run() depending on your storage format
    print("Done. Implement save logic in this script.")

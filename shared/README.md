# shared/

Utilities shared across all tasks. Import from here instead of duplicating code.

| File | What it provides |
|---|---|
| `seed.py` | `set_seed(n)` — fix all random seeds (Python, NumPy, PyTorch, HuggingFace) |
| `config.py` | `load_config(path, overrides)` — load YAML with optional CLI overrides |
| `paths.py` | `data_dir(__file__)`, `config_dir(__file__)` — paths relative to the script |
| `calt_adapter.py` | Re-exports `DatasetPipeline`, `IOPipeline`, `ModelPipeline`, `TrainerPipeline` |
| `logging.py` | `CustomLoggingCallback` — logs gradient norms and GPU memory to wandb |
| `plotting.py` | `plot_success_rate()`, `show_examples()` — for analysis notebooks |

## Usage example

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from shared import set_seed, load_config
from shared.paths import data_dir, config_dir

set_seed(42)
cfg = load_config(config_dir(__file__) / "train.yaml")
```

# [TASK NAME]

**Task**: [One sentence: "Given X, predict Y."]

## The math

[Explain the mathematical problem here. What is the input? What is the output?
What algorithm computes the answer? Why is it non-trivial for a neural network?]

## Data format

Input  : [describe the string representation of the input]
Output : [describe the string representation of the output]

Example:
    "[example input]"  →  "[example output]"

## Quick start

```bash
cd [task_name]/experiments/toy/scripts
python generate.py    # generate dataset
python train.py       # train model
python evaluate.py    # report success rate
```

---

## Checklist for creating a new task

- [ ] Fill in `core/generator.py` — implement `TaskGenerator.__call__(seed) → (problem, answer)`
- [ ] Fill in `core/formatter.py` — define string format for problem and answer
- [ ] Optional: fill in `core/parser.py` and enable it in `core/train.py` — only for JSON Lines data or load-time transformations; plain text needs no parser
- [ ] Optional: implement `instance_stats` in `core/metrics.py` and pass it to `DatasetPipeline` in `scripts/generate.py`
- [ ] Replace `task_template` in the imports of `experiments/toy/scripts/*.py` with your task's folder name
- [ ] Update `experiments/toy/configs/data.yaml` — `problem_generator` parameters, dataset size, seed, backend
- [ ] Update `experiments/toy/configs/lexer.yaml` — token vocabulary for your task
- [ ] Update `experiments/toy/configs/train.yaml` — model and training hyperparameters
- [ ] Update this README

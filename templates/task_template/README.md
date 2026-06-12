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

- [ ] Fill in `core/generator.py` — implement `sample_instance(seed) → (problem, answer)`
- [ ] Fill in `core/formatter.py` — define string format for problem and answer
- [ ] Fill in `core/parser.py` — implement `process_sample` for the load preprocessor
- [ ] Fill in `core/metrics.py` — implement `instance_stats` and `success_rate`
- [ ] Update `experiments/toy/configs/data.yaml` — dataset size, seed, backend
- [ ] Update `experiments/toy/configs/lexer.yaml` — token vocabulary for your task
- [ ] Update `experiments/toy/configs/train.yaml` — model and training hyperparameters
- [ ] Update this README

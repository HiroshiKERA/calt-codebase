# Border basis — finite_field experiment

**Goal**: vary the size of the base finite field GF(p) and observe how the border
basis structure changes with p.

The prime `p` is chosen with the `--p` flag (any prime; the configs default to
GF(7)).

## How to run
```bash
cd border_basis/experiments/finite_field/scripts
python generate.py --p 5         # → ../data/GF5/
python train.py --p 5
```

Each `p` writes its dataset to `../data/GF{p}/` and the trained model to
`../outputs/results_GF{p}/`, so runs do not clash.

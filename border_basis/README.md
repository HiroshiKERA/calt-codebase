# border_basis

**Task**: Given polynomial generators of a zero-dimensional ideal I ⊂ K[x, y],
predict the border basis of I.

## The math

### What is a border basis?

Let I ⊂ K[x1,…,xn] be a **zero-dimensional ideal** (the variety V(I) is a finite set
of points). A **border basis** of I is an alternative to a Gröbner basis that is:

- Uniquely determined by a chosen **order ideal** O(B) (a set of monomials
  closed under divisibility that indexes the quotient ring K[x]/I)
- Often **more numerically stable** than a Gröbner basis
- Related to the **BM algorithm** (Mourrain / Cerlienco–Mureddu)

**Formal definition**:
Let O = {t1, …, tμ} be an order ideal for I of size μ = dimK(K[x]/I).
The **border** ∂O = {x·ti : ti ∈ O, x·ti ∉ O} of O consists of the monomials
just outside O. The border basis B = {b_τ : τ ∈ ∂O} satisfies:

    b_τ = τ − Σ_{ti ∈ O} c_{τ,i} · ti

i.e., each border monomial τ reduces to a linear combination of O-monomials.

### Key difference from Gröbner basis

| | Gröbner basis | Border basis |
|---|---|---|
| Ideal type | Any ideal | Zero-dimensional ideals |
| Monomial order | Required (lex, grevlex, …) | Not required |
| Numerical stability | Often poor | Often better |
| Uniqueness | Up to choice of order | Up to choice of order ideal |

### This task

- **Input**: polynomial generators g1, …, gk over GF(p) (a finite field)
- **Output**: the elements of the border basis of ⟨g1,…,gk⟩

Over GF(p), three random generators in two variables give a zero-dimensional
ideal with high probability; the generator checks `ideal.dimension() == 0` and
redraws (up to 100 times per seed) otherwise.

## Status

✅ **Real border basis algorithm implemented** in `core/algorithm.py`. It is a
verbatim port of Algorithm 4.1 (BBasis) from:

> Kera, Pelleriti, Ishihara, Zimmer, Pokutta,
> *Computational Algebra with Attention: Transformer Oracles for Border Basis Algorithms*,
> arXiv:2505.23696, 2025.
> Code: <https://github.com/HiroshiKERA/OracleBorderBasis>

The algorithm:
1. Computes the **L-stable span** of F up to a sufficient degree (Algorithm 2.13).
2. Finds the **optimal order ideal** O via a Mixed Integer Linear Program
   (Lemma 3.9, solved with `scipy.optimize.milp`).
3. Applies a **basis transformation** (Algorithm 4.3) to extract the border basis G.

Fast Gaussian elimination uses a `SortedList` from `sortedcontainers` for
O(log n) reducer lookup.

## Experiments

| Experiment | What varies |
|---|---|
| `toy/` | Small field GF(7), 2 variables, small degree |
| `finite_field/` | GF(p) for any prime p chosen with `--p` (the configs default to GF(7)) |
| `ablation/` | Model size (small / medium / large) on the toy GF(7) data |

## Quick start

```bash
cd border_basis/experiments/toy/scripts
python generate.py
python train.py
python evaluate.py
```

## User post-processing hooks (optional)

Two opt-in slots inject custom `(input, target)` transformations between the
load chain and tokenization. Identical mechanism to `groebner_basis`, see
[groebner_basis/README.md](../groebner_basis/README.md#user-post-processing-hooks-optional).

Per-task slot for this task: `border_basis/core/postprocessor.py` (copy the
`.example` file and edit).

## Tokenization format (raw vs C/E expanded)

Same mechanism as `groebner_basis`. Two lexer configs are provided and the format is
auto-detected from `data.lexer_config` in your `train.yaml`:

```yaml
data:
  lexer_config: ../configs/lexer.yaml           # raw  (default, ISSAC '26 paper)
  # lexer_config: ../configs/lexer_expanded.yaml  # C/E expanded (smaller vocab)
```

- **raw** — `"x^2 + y"` → `["x", "^", "2", "+", "y"]`. Default.
- **expanded** — each term as `C<coef> E<exp_x> E<exp_y>`. Fixed-size vocab.

When `expanded` is detected, `core/train.py` wires `ExpandedFormLoadPreprocessor` into
the load chain, which needs the source ring from `data.yaml`. `train.py` reads
`../configs/data.yaml` automatically, so pass `--data_config_path` only if your data
config lives elsewhere. See [groebner_basis/README.md](../groebner_basis/README.md#tokenization-format-raw-vs-ce-expanded)
for details and trade-offs (they are identical for both tasks).

# border_basis

**Task**: Given polynomial generators of a zero-dimensional ideal I ⊂ K[x, y, z],
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

Working over GF(p) ensures the ideal is automatically zero-dimensional
(the variety V(I) ⊆ GF(p)^n is a finite set of at most p^n points).

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
| `finite_field/` | Compare GF(7), GF(11), GF(17), GF(31) |
| `ablation/` | Number of generators, degree bounds |

## Quick start (once TODO is filled)

```bash
cd border_basis/experiments/toy/scripts
python generate.py
python train.py
python evaluate.py
```

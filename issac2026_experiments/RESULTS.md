# Table 8 results — standard vs monomial token embedding

All runs: 32 epochs, identical expanded-form data, 3 seeds (42, 123, 7).
Success rate is the exact-match accuracy of autoregressive greedy generation.
Peak GPU memory is the maximum resident memory of the training process, sampled every 15 s.

| Task | Field | Standard | Monomial | Peak mem std | Peak mem mono |
|---|---|---|---|---|---|
| P-Multiplication | Z | 8.0 ± 0.9 | n/a | 3.9 GB | — |
| P-Multiplication | F7 | 1.0 ± 0.2 | 1.1 ± 0.1 | 3.4 GB | 1.6 GB |
| P-Multiplication | F31 | 0.2 ± 0.1 | 0.2 ± 0.1 | 3.5 GB | 1.6 GB |
| P-Multiplication | F97 | 0.0 ± 0.1 | 0.1 ± 0.1 | 3.7 GB | 1.7 GB |
| P-Multiplication+ | Z | 75.2 ± 1.3 | n/a | 9.3 GB | — |
| P-Multiplication+ | F7 | 52.0 ± 0.9 | 79.4 ± 0.1 | 7.8 GB | 2.3 GB |
| P-Multiplication+ | F31 | 0.0 ± 0.0 | 52.4 ± 0.2 | 7.5 GB | 2.2 GB |
| P-Multiplication+ | F97 | 0.0 ± 0.0 | 9.7 ± 15.0 | 8.3 GB | 2.3 GB |
| P-Reduction | Z | 77.1 ± 0.5 | n/a | 1.7 GB | — |
| P-Reduction | F7 | 24.9 ± 19.8 | 2.7 ± 0.4 | 1.6 GB | 1.4 GB |
| P-Reduction | F31 | 0.3 ± 0.2 | 0.4 ± 0.3 | 1.6 GB | 1.4 GB |
| P-Reduction | F97 | 0.2 ± 0.1 | 0.0 ± 0.1 | 1.5 GB | 1.4 GB |
| P-Gröbner | F7 | 44.2 ± 1.3 | 44.9 ± 2.2 | 11.9 GB | 3.6 GB |

## Per-seed values (42 / 123 / 7)

| Run | Values |
|---|---|
| `gb_repr_monomial` | 42.3 / 45.9 / 46.4 |
| `gb_repr_standard` | 44.9 / 42.7 / 44.9 |
| `pm_GF31_monomial_full` | 52.3 / 52.3 / 52.6 |
| `pm_GF31_monomial_last_element` | 0.2 / 0.3 / 0.2 |
| `pm_GF31_standard_full` | 0.0 / 0.0 / 0.0 |
| `pm_GF31_standard_last_element` | 0.2 / 0.1 / 0.2 |
| `pm_GF7_monomial_full` | 79.4 / 79.4 / 79.3 |
| `pm_GF7_monomial_last_element` | 1.1 / 1.1 / 1.2 |
| `pm_GF7_standard_full` | 52.8 / 51.0 / 52.2 |
| `pm_GF7_standard_last_element` | 1.2 / 0.8 / 1.0 |
| `pm_GF97_monomial_full` | 2.1 / 26.9 / 0.0 |
| `pm_GF97_monomial_last_element` | 0.2 / 0.0 / 0.0 |
| `pm_GF97_standard_full` | 0.0 / 0.0 / 0.0 |
| `pm_GF97_standard_last_element` | 0.0 / 0.0 / 0.1 |
| `pm_ZZ_standard_full` | 74.1 / 74.9 / 76.6 |
| `pm_ZZ_standard_last_element` | 8.9 / 7.2 / 7.9 |
| `pr_GF31_monomial` | 0.7 / 0.4 / 0.2 |
| `pr_GF31_standard` | 0.5 / 0.2 / 0.1 |
| `pr_GF7_monomial` | 3.1 / 2.3 / 2.8 |
| `pr_GF7_standard` | 2.1 / 36.2 / 36.5 |
| `pr_GF97_monomial` | 0.1 / 0.0 / 0.0 |
| `pr_GF97_standard` | 0.2 / 0.1 / 0.3 |
| `pr_ZZ_standard` | 76.7 / 77.6 / 77.0 |

## Notes

- **Z + monomial is not applicable.** The Z lexer uses `digit_group: 3` and coefficients reach ~19,000,
  so a single coefficient spans several tokens and the monomial embedding cannot keep its fixed
  monomial width (1 coefficient + n exponents + 1 separator).
- **P-Reduction over F7 (standard) is strongly seed-dependent**: 2.1 / 36.2 / 36.5 % across the three
  seeds. Reporting a single seed for this cell is misleading.
- **P-Reduction requires the zero-polynomial fix in the library.** `poly_to_expanded_form` serialized
  the zero polynomial as a bare `C0`, without exponent slots, which breaks the monomial alignment;
  17 % of P-Reduction targets have a zero remainder. Fixed by emitting `C0 E0 ... E0`.
  The P-Reduction rows above were produced with that fix.

## Table 7 — Gröbner basis degree and term count

| Setting | Mean degree (lex) | Mean degree (degrevlex) | Mean # terms (lex) | Mean # terms (degrevlex) |
|---|---|---|---|---|
| GF7, 2 vars, deg<=4 (n=1000) | 3.21 | 2.41 | 6.50 | 6.90 |
| QQ, 2 vars, deg<=4 (n=1000) | 2.52 | 2.07 | 4.34 | 3.98 |
| GF7, 2 vars, deg<=16 (n=200) | 26.33 | 7.90 | 36.13 | 24.36 |
| GF7, 2 vars, deg<=32 (n=200) | 72.71 | 16.70 | 105.17 | 132.30 |
| GF7, 5 vars, deg<=4, 5 polys (n=200) | 10.28 | 2.91 | 139.76 | 95.96 |

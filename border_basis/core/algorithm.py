"""
Border basis algorithm (Algorithm 4.1 BBasis) from
*Computational Algebra with Attention: Transformer Oracles for Border Basis Algorithms*
(Kera, Pelleriti, Ishihara, Zimmer, Pokutta — arXiv:2505.23696, 2025).

This file is a verbatim port of
`HiroshiKERA/OracleBorderBasis/src/border_basis_lib/border_basis.py`.

License: see https://github.com/HiroshiKERA/OracleBorderBasis (MIT)
"""

# ruff: noqa
# fmt: off
from sage.all import *
from typing import List, Tuple
from pprint import pprint
from time import time

# for faster computation
from sortedcontainers import SortedList


class BorderBasisCalculator:
    def __init__(self, ring):
        """Initialize with a polynomial ring."""
        self.ring = ring
        self.variables = ring.gens()

        # sorting time
        self.sorting_time = 0

    def terms_up_to_degree(self, d: int) -> List:
        """Compute all terms up to degree d."""
        n = len(self.variables)
        terms = [self.ring(1)]
        for t in range(1, d + 1):
            exponents = list(WeightedIntegerVectors(t, [1] * n))
            terms.extend([self.ring.monomial(*e) for e in exponents])
        return terms

    def is_all_divisors_in(self, t, O: List) -> bool:
        """Check if adding term t to order ideal O maintains the order ideal property."""
        if t.degree() in (0, 1):
            return True
        return all([xi in O and t / xi in O for xi in self.variables if xi.divides(t)])

    def border(self, O: List) -> List:
        """Compute the border of an order ideal O."""
        border = list()
        for t in O:
            for var in self.variables:
                new_term = t * var
                if new_term not in O and new_term not in border:
                    border.append(new_term)
        return border

    def extend_V(self, V: List) -> List:
        V_plus = []
        for v in V:
            for var in self.variables:
                new_term = v * var
                V_plus.append(new_term)
        return V_plus

    def compute_border_basis(self, F: List, weights: dict, use_fast_elimination=False, lstabilization_only=False) -> Tuple[List, List]:
        """Algorithm 4.1 (BBasis)."""
        timings = {}

        s = time()
        d = max(f.degree() for f in F)
        timings['step1_initial_degree'] = time() - s

        while True:
            s = time()
            L = self.terms_up_to_degree(d)
            M = self.compute_lstable_span(F, L, use_fast_elimination=use_fast_elimination)
            timings['step2_lstable_span'] = time() - s

            s = time()
            terms_d = [t for t in L if t.degree() == d]
            lt_M = set(f.lm() for f in M)
            sufficient_universe = all(t in lt_M for t in terms_d)
            timings['step3_check_universe'] = time() - s

            if not sufficient_universe:
                d += 1
                continue

            s = time()
            d_old = d
            d = len(self.terms_up_to_degree(d)) - len(M)
            if d <= d_old:
                M = [f for f in M if f.degree() <= d]
            else:
                M = self.compute_lstable_span(F, self.terms_up_to_degree(d), use_fast_elimination=use_fast_elimination)
            timings['step4_adjust_dimension'] = time() - s
            break

        if not lstabilization_only:
            s = time()
            O = self.find_optimal_order_ideal(M, d, weights)
            timings['step5_find_order_ideal'] = time() - s

            s = time()
            G = self.basis_transformation(M, O, use_fast_elimination=use_fast_elimination)
            timings['step6_compute_basis'] = time() - s

            timings['total_time'] = sum(timings.values())
        else:
            O = []
            G = M

        return G, O, timings

    def basis_transformation(self, M: List, O: List, use_fast_elimination=False) -> List:
        """Algorithm 4.3 (BasisTransformation)."""
        if not M:
            return []

        max_deg = max(p.degree() for p in M)

        def term_key(p):
            lt = p.lm()
            return (lt not in O, lt)

        M_sorted = sorted(M, key=term_key)
        # gaussian_elimination returns (W, ...) — unpack the polynomial list.
        Gprime = self.gaussian_elimination([], M_sorted, use_fast_elimination=use_fast_elimination)[0]

        border = self.border(O)
        G = [g for g in Gprime if g.lm() in border]

        return G

    def gaussian_elimination_fast(self, V: List, G: List) -> List:
        """Optimized Gaussian elimination using SortedList + binary search."""
        assert (all([v.lc() == 1 for v in V]))

        reducers = SortedList(V, key=lambda p: p.lm())
        W = []
        non_zero_reductions_indices = []

        from collections import defaultdict
        reduction_indices = defaultdict(list)
        reducer_indices = {}

        for idx, f in enumerate(G):
            if f == 0:
                continue

            reduced = True
            while reduced and f != 0:
                reduced = False
                f_lead_term = f.lm()

                index = reducers.bisect_key_left(f_lead_term)
                if index < len(reducers) and reducers[index].lm() == f_lead_term:
                    reducer = reducers[index]
                    f = f - f.lc() * reducer
                    reduced = True
                    if index >= len(V):
                        reduction_indices[idx].append(reducer_indices[index])

            if f != 0:
                f = f / f.lc()
                reducers.add(f)
                W.append(f)
                non_zero_reductions_indices.append(idx)
                reducer_indices[len(reducers) - 1] = idx
                if not idx in reduction_indices.keys():
                    reduction_indices[idx] = []

        return W, non_zero_reductions_indices, reduction_indices

    def gaussian_elimination(self, V: List, G: List, use_fast_elimination=False) -> List:
        """Algorithm 2.11 (GaussEl)."""
        assert (all([v.lc() == 1 for v in V]))

        if use_fast_elimination:
            s = time()
            V = sorted(V, key=lambda p: p.lm())
            self.sorting_time += time() - s
            return self.gaussian_elimination_fast(V, G)

        reducers = list(V)
        H = list(G)

        while H:
            f = H.pop(0)
            if f == 0:
                continue

            reduced = True
            while reduced and f != 0:
                reduced = False
                for reducer in reducers:
                    if reducer.lm() == f.lm():
                        f = f - f.lc() * reducer
                        reduced = True
                        break

            if f != 0:
                f = f / f.lc()
                reducers.append(f)

        W = reducers[len(V):]
        return W, None

    def compute_lstable_span(self, F: List, L: List, use_fast_elimination=False) -> List:
        """Algorithm 2.13 (LStabSpan)."""
        d = max(l.degree() for l in L)
        # gaussian_elimination returns (W, ...) — unpack the polynomial list.
        V = self.gaussian_elimination([], F, use_fast_elimination=use_fast_elimination)[0]

        while True:
            W = self.gaussian_elimination(V, self.border(V), use_fast_elimination=use_fast_elimination)[0]
            W = [w for w in W if w.degree() <= d]

            if W:
                V.extend(W)
            else:
                break

        return V

    def find_optimal_order_ideal(self, M: List, d: int, weights: dict) -> List:
        """Lemma 3.9: alternative description of order ideal polytope (MILP)."""
        from scipy.optimize import milp, Bounds, LinearConstraint
        from itertools import combinations
        import numpy as np

        terms = self.terms_up_to_degree(d - 1)
        n_terms = len(terms)

        c = np.array([-weights.get(term, 0) for term in terms])

        integrality = np.ones(n_terms, dtype=np.int32)
        bounds = Bounds(0, 1)

        constraints = []

        # Constraint (3.6a): Order ideal property
        order_rows = []
        for i, t1 in enumerate(terms):
            for j, t2 in enumerate(terms):
                if t1.divides(t2) and t1 != t2:
                    row = np.zeros(n_terms)
                    row[i] = -1
                    row[j] = 1
                    order_rows.append(row)

        if order_rows:
            A_order = np.vstack(order_rows)
            constraints.append(LinearConstraint(A_order, -np.inf, 0))

        # Constraint (3.6b): Size of order ideal is d
        A_eq = np.ones((1, n_terms))
        constraints.append(LinearConstraint(A_eq, d, d))

        M_d_minus_1 = [f for f in M if f.degree() <= d - 1]
        M_size = len(M_d_minus_1)

        coeff_matrix = np.zeros((len(M_d_minus_1), len(terms)))
        for i, f in enumerate(M_d_minus_1):
            for j, term in enumerate(terms):
                coeff = f.monomial_coefficient(term)
                if coeff != 0:
                    coeff_matrix[i, j] = float(coeff)

        def get_submatrix_rank(U_indices):
            return np.linalg.matrix_rank(coeff_matrix[:, U_indices])

        # Constraint (3.6c)
        rank_rows = []
        rank_lbs = []

        for U_indices in combinations(range(n_terms), M_size):
            rank_U = get_submatrix_rank(U_indices)
            row = np.zeros(n_terms)
            row[list(U_indices)] = 1
            rank_rows.append(row)
            rank_lbs.append(M_size - rank_U)

        if rank_rows:
            A_rank = np.vstack(rank_rows)
            constraints.append(LinearConstraint(A_rank, rank_lbs, np.inf))

        options = {
            'disp': False,
            'presolve': True,
            'mip_rel_gap': 1e-4,
        }

        result = milp(
            c=c,
            integrality=integrality,
            bounds=bounds,
            constraints=constraints,
            options=options,
        )

        if not result.success:
            raise ValueError(f"MILP failed: {result.message}")

        O = list(set(term for i, term in enumerate(terms) if result.x[i] > 0.5))
        O = sorted(O, key=lambda t: t)
        return O

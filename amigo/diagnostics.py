"""
amigo.diagnostics
=================
General-purpose pre-optimisation diagnostic checks for any Amigo model.

Usage
-----
::

    import amigo as am

    diag = am.Diagnostics(model, x, lower, upper)
    diag.run(
        spotlights={
            "ic.res": ["velocity", "gamma", "altitude", "range", "mass"],
            "fc.res": ["velocity", "gamma", "altitude"],
        },
    )

The five checks
---------------
1. **NaN/Inf** - scan every entry of the design vector.
2. **Bounds**  - compare x against user-supplied lower/upper vectors and
   against component-declared meta bounds; flag zero-valued bounds that were
   never set (``create_vector()`` defaults to 0).
3. **Connectivity** - variables sharing the same short name across different
   components but whose global indices don't overlap are likely missing a
   ``model.link()`` call.
4. **Constraint residuals** - evaluate all constraints using
   ``model.eval_constraints(x)`` and report ``max|res|`` / ``mean|res|`` for
   each group.  The largest value is an estimate of ``inf_pr`` at iter 0.
5. **Spotlight constraints** - read named constraint-residual variables
   directly from ``x`` (useful for IC/FC whose values the user seeds during
   initial-guess setup) and report per-element pass/fail.
"""

from __future__ import annotations

import numpy as np
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid circular import at runtime
    from .model import Model, ModelVector


# ── helpers ──────────────────────────────────────────────────────────────────

_SEP = "═" * 62


def _strip_indices(expr: str) -> str:
    """Remove bracket subscript from 'comp.var[0]' → 'comp.var'."""
    return expr.split("[")[0].strip()


# ── main class ────────────────────────────────────────────────────────────────


class Diagnostics:
    """Pre-optimisation health checks for an initialised Amigo model.

    Parameters
    ----------
    model:
        An initialised ``am.Model`` instance.
    x:
        Design vector returned by ``model.create_vector()`` and populated
        with the initial guess.
    lower, upper:
        Optional bound vectors returned by ``model.create_vector()``.
        When omitted, only the component-declared meta bounds are checked.
    """

    _TOL_BOUND: float = 1e-10
    _MAX_VIOLATIONS_SHOWN: int = 5

    def __init__(self, model, x, lower=None, upper=None):
        self.model = model
        self.x = x
        self.lower = lower
        self.upper = upper
        self._idx_to_name: dict[int, str] = self._build_idx_map()

    # ── public API ────────────────────────────────────────────────────────────

    def run(
        self,
        spotlights: dict | None = None,
        tol: float = 1e-6,
    ):
        """Run all five checks and print a formatted report.

        Parameters
        ----------
        spotlights:
            ``{constraint_var_name: [label_0, label_1, ...]}``.
            Each named constraint variable is read directly from ``x`` and
            its per-element value is printed with a pass/fail tag.  Useful
            when the user explicitly seeds constraint residual variables (e.g.
            IC/FC targets) as part of the initial guess.
        tol:
            Residual tolerance used for pass/fail in spotlight check.
        """
        spotlights = spotlights or {}

        nvar = self.model.num_variables
        ncon = self.model.num_constraints
        mname = getattr(self.model, "module_name", None) or "?"

        print(f"\n{_SEP}")
        print(f"  Amigo Diagnostics")
        print(f"  Model: {mname!r}   Variables: {nvar}   Constraints: {ncon}")
        print(_SEP)

        nan_fail = self._check_nan_inf()
        bounds_fail = self._check_bounds()
        connect_fail = self._check_connectivity()
        constrain_fail = self._check_constraints()
        spot_fail = self._check_spotlights(spotlights, tol)

        print(f"{_SEP}\n")

        total_fail = (
            nan_fail or bounds_fail or connect_fail or constrain_fail or spot_fail
        )

        fail_dict = {
            "nan_inf": nan_fail,
            "bounds": bounds_fail,
            "connectivity": connect_fail,
            "constraints": constrain_fail,
            "spotlights": spot_fail,
        }

        return total_fail, fail_dict

    # ── check 1: NaN / Inf ────────────────────────────────────────────────────

    def _check_nan_inf(self) -> bool:
        fail = True
        print("\n  [1] NaN / Inf in design vector")
        x_arr = np.asarray(self.x[:], dtype=float)
        bad = ~np.isfinite(x_arr)
        n_bad = int(bad.sum())
        if n_bad == 0:
            print(f"      [OK]   No NaN/Inf ({x_arr.size} entries checked)")
            fail = False
        else:
            print(f"      [FAIL] {n_bad} non-finite entries")
            for idx in np.where(bad)[0][: self._MAX_VIOLATIONS_SHOWN]:
                name = self._idx_to_name.get(int(idx), f"idx={idx}")
                print(f"             {name:40s}  val={x_arr[idx]}")

        return fail

    # ── check 2: bounds ───────────────────────────────────────────────────────

    def _check_bounds(self) -> bool:
        fail = False
        print("\n  [2] Bounds analysis")

        x_arr = np.asarray(self.x[:], dtype=float)

        # --- user-supplied bound vectors ---
        if self.lower is not None and self.upper is not None:
            lo_arr = np.asarray(self.lower[:], dtype=float)
            hi_arr = np.asarray(self.upper[:], dtype=float)

            lo_viol_mask = x_arr < lo_arr - self._TOL_BOUND
            hi_viol_mask = x_arr > hi_arr + self._TOL_BOUND
            lo_n = int(lo_viol_mask.sum())
            hi_n = int(hi_viol_mask.sum())

            if lo_n == 0 and hi_n == 0:
                print("      [OK]   All variables within supplied bounds")
            else:
                fail = True
                print(f"      [FAIL] Bounds violations: {lo_n} lower, {hi_n} upper")
                self._print_violations(
                    "lower", lo_viol_mask, lo_arr - x_arr, lo_arr, x_arr
                )
                self._print_violations(
                    "upper", hi_viol_mask, x_arr - hi_arr, hi_arr, x_arr
                )
                total = lo_n + hi_n
                if total > self._MAX_VIOLATIONS_SHOWN:
                    print(f"             ... ({total} total violations)")
        else:
            print("      [SKIP] No user bound vectors supplied")

        # --- meta-declared bounds vs user vector ---
        meta_fail = self._check_meta_bounds(x_arr)

        return fail or meta_fail

    def _print_violations(
        self,
        side: str,
        mask: np.ndarray,
        gap: np.ndarray,
        bound: np.ndarray,
        x_arr: np.ndarray,
    ) -> None:
        # Retrieve meta-declared bounds for comparison
        try:
            meta_key = "lower" if side == "lower" else "upper"
            meta_vec = np.asarray(
                self.model.get_values_from_meta(meta_key)[:], dtype=float
            )
        except Exception:
            meta_vec = None

        indices = np.where(mask)[0]
        worst = indices[np.argsort(gap[indices])[::-1]][: self._MAX_VIOLATIONS_SHOWN]
        for idx in worst:
            name = self._idx_to_name.get(int(idx), f"idx={idx}")
            bval = bound[idx]
            xval = x_arr[idx]
            note = "  [unset?]" if bval == 0.0 else ""
            meta_note = ""
            if meta_vec is not None and np.isfinite(meta_vec[idx]):
                meta_note = f"  [component declared: {meta_vec[idx]:.4g}]"
            print(
                f"             {side} {name:35s}  "
                f"x={xval:.4g}  bound={bval:.4g}{note}{meta_note}"
            )

    def _check_meta_bounds(self, x_arr: np.ndarray) -> bool:
        """Warn about *input* variables where the component declared a finite,
        non-zero bound but the user's bound vector has 0 (never set).

        Constraint variables are skipped — their meta upper=0 is intentional
        (Amigo enforces equality constraints as c(x)=0).
        """
        fail = False

        if self.upper is None or self.lower is None:
            return

        # Only examine input variables; constraint variables are equality
        # constraints whose upper/lower = 0 is correct by design.
        input_names, _, _, _ = self.model.get_names()

        issues = []
        for side, meta_key, user_vec_attr in (
            ("upper", "upper", "upper"),
            ("lower", "lower", "lower"),
        ):
            try:
                meta_vec_model = self.model.get_values_from_meta(meta_key)
            except Exception:
                continue
            user_vec = np.asarray(getattr(self, user_vec_attr)[:], dtype=float)

            for vname in input_names:
                try:
                    idxs = np.atleast_1d(self.model.get_indices(vname)).flatten()
                    meta_vals = np.asarray(meta_vec_model[vname], dtype=float).flatten()
                except Exception:
                    continue
                for local_i, (idx, mv) in enumerate(zip(idxs, meta_vals)):
                    uv = user_vec[int(idx)]
                    # Flag only when the component declared a *meaningful* finite
                    # bound (non-zero, non-inf) but the user vector is still 0.
                    if np.isfinite(mv) and mv != 0.0 and uv == 0.0:
                        label = f"{vname}[{local_i}]" if len(idxs) > 1 else vname
                        issues.append(
                            f"             {side} {label:40s}  "
                            f"component declared {mv:.4g}, user vector = 0 [unset?]"
                        )

        if issues:
            fail = True
            print(
                "      [WARN] Component-declared bounds not reflected in user vectors:"
            )
            for line in issues[: self._MAX_VIOLATIONS_SHOWN]:
                print(line)
            if len(issues) > self._MAX_VIOLATIONS_SHOWN:
                print(f"             ... ({len(issues)} total)")
        else:
            print("      [OK]   Component meta bounds are consistent with user vectors")

        return fail

    # ── check 3: connectivity ─────────────────────────────────────────────────

    def _check_connectivity(self) -> bool:
        fail = False
        print("\n  [3] Connectivity (potentially missing model.link() calls)")

        inputs, _, _, _ = self.model.get_names()

        # Variables that appear as source or target in any link
        linked_src: set[str] = set()
        linked_tgt: set[str] = set()
        for src_expr, _, tgt_expr, _ in self.model.links:
            linked_src.add(_strip_indices(src_expr))
            linked_tgt.add(_strip_indices(tgt_expr))

        # Group inputs by their short name (last segment)
        short_map: dict[str, list[str]] = defaultdict(list)
        for full in inputs:
            short = full.rsplit(".", 1)[-1]
            short_map[short].append(full)

        warnings: list[str] = []
        for short, full_names in short_map.items():
            if len(full_names) < 2:
                continue
            # Check whether any pair shares global indices (i.e. is linked)
            try:
                idx_sets = [
                    set(np.atleast_1d(self.model.get_indices(n)).tolist())
                    for n in full_names
                ]
            except Exception:
                continue
            # If NO pair shares indices, none of them are linked together
            any_shared = any(
                len(idx_sets[i] & idx_sets[j]) > 0
                for i in range(len(idx_sets))
                for j in range(i + 1, len(idx_sets))
            )
            if not any_shared:
                warnings.append(
                    f"      [WARN] '{short}' appears in {full_names} "
                    f"but indices don't overlap — missing link?"
                )

        if warnings:
            fail = True
            for w in warnings:
                print(w)
        else:
            print("      [OK]   No shared-name unlinked variables detected")

        # Isolated inputs (not in any link at all)
        isolated = [
            n
            for n in inputs
            if _strip_indices(n) not in linked_src
            and _strip_indices(n) not in linked_tgt
        ]
        if isolated:
            print(
                f"      [INFO] {len(isolated)} input(s) not in any link "
                f"(may be intentional standalone design variables):"
            )
            for name in isolated[: self._MAX_VIOLATIONS_SHOWN]:
                print(f"             {name}")
            if len(isolated) > self._MAX_VIOLATIONS_SHOWN:
                print(f"             ... ({len(isolated)} total)")

        return fail

    # ── check 4: constraint residuals via model.eval_constraints ─────────────

    def _check_constraints(self) -> bool:
        fail = False
        print("\n  [4] Constraint residuals (model.eval_constraints)")
        try:
            residuals = self.model.eval_constraints(self.x)
        except Exception as exc:
            print(f"      [ERR]  eval_constraints failed: {exc}")
            fail = True
            return fail

        all_max: list[float] = []
        for name, arr in residuals.items():
            arr = np.asarray(arr, dtype=float).flatten()
            if arr.size == 0:
                continue
            max_r = float(np.max(np.abs(arr)))
            mean_r = float(np.mean(np.abs(arr)))
            all_max.append(max_r)
            flag = "  [!!]" if max_r > 1.0 else "      "
            print(f"  {flag}  {name:38s}  max={max_r:.4e}  mean={mean_r:.4e}")

        if all_max:
            inf_pr = max(all_max)
            tag = "[WARN]" if inf_pr > 1.0 else "[OK]  "
            print(
                f"\n      {tag} inf_pr estimate = {inf_pr:.4e}"
                f"  (should match optimizer iter-0 inf_pr)"
            )
            if inf_pr > 1.0:
                fail = True

        return fail

    # ── check 5: spotlight constraints ───────────────────────────────────────

    def _check_spotlights(self, spotlights: dict, tol: float) -> bool:
        fail = False
        print("\n  [5] Spotlight constraints (read from x)")

        if not spotlights:
            print("      [SKIP] No spotlights specified.")
            return

        for cons_name, labels in spotlights.items():
            try:
                vals = np.asarray(self.x[cons_name], dtype=float).flatten()
            except Exception as exc:
                print(f"      [ERR]  {cons_name}: {exc}")
                fail = True
                continue
            print(f"      {cons_name}:")
            for label, v in zip(labels, vals):
                tag = "[OK]  " if abs(v) < tol else "[FAIL]"
                print(f"        {tag} {label:14s}  val={v:.4e}")
                if abs(v) < tol:
                    fail = True

        return fail

    # ── internal helpers ──────────────────────────────────────────────────────

    def _build_idx_map(self) -> dict[int, str]:
        """Map each global variable index to a human-readable name."""
        idx_to_name: dict[int, str] = {}
        inputs, cons, _, _ = self.model.get_names()
        for vname in inputs + cons:
            try:
                idxs = np.atleast_1d(self.model.get_indices(vname))
                for i, idx in enumerate(idxs):
                    idx_to_name[int(idx)] = f"{vname}[{i}]"
            except Exception:
                pass
        return idx_to_name

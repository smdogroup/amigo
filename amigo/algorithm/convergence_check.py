"""Convergence tests for the interior-point loop.

Four criteria are checked each iteration.  Primary convergence
requires every KKT component below its tolerance.  Divergence halts
the solve when the iterate magnitude exceeds a safety bound.
Acceptable convergence flags a weaker solution once relaxed
tolerances hold for several iterations in a row.  Precision-floor
detection catches bit-identical residuals, the numerical limit below
which further progress is not possible.
"""

import numpy as np

# Return codes for convergence check
CONTINUE = 0
CONVERGED = 1
CONVERGED_ACCEPTABLE = 2
DIVERGED = 3
PRECISION_FLOOR = 4


class ConvergenceCheck:
    """Convergence checks: primary, acceptable, divergence, precision floor."""

    def _check_convergence(
        self,
        i,
        options,
        mult_ind,
        res_norm,
        prev_res_norm,
        acceptable_counter,
        precision_floor_count,
        comm_rank,
    ):
        """Check all convergence criteria.

        Returns (status, overall_error, acceptable_counter, precision_floor_count)
        where status is one of the module-level constants.
        """
        tol = options["convergence_tolerance"]
        dual_inf_tol = options["dual_inf_tol"]
        constr_viol_tol = options["constr_viol_tol"]
        compl_inf_tol = options["compl_inf_tol"]
        diverging_iterates_tol = options["diverging_iterates_tol"]
        acceptable_tol = options["acceptable_tol"]
        acceptable_iter = options["acceptable_iter"]
        acceptable_dual_inf_tol = options["acceptable_dual_inf_tol"]
        acceptable_constr_viol_tol = options["acceptable_constr_viol_tol"]
        acceptable_compl_inf_tol = options["acceptable_compl_inf_tol"]

        # Compute NLP error components at mu_target=0
        d_inf_nlp, p_inf_nlp, c_inf_nlp = self.optimizer.compute_kkt_error_mu(
            0.0, self.vars, self.grad
        )
        s_d_conv, s_c_conv = self._compute_optimality_scaling(mult_ind)
        overall_error = max(d_inf_nlp / s_d_conv, p_inf_nlp, c_inf_nlp / s_c_conv)

        # Primary convergence: ALL 4 conditions must hold
        if (
            overall_error <= tol
            and d_inf_nlp <= dual_inf_tol
            and p_inf_nlp <= constr_viol_tol
            and c_inf_nlp <= compl_inf_tol
        ):
            if comm_rank == 0:
                obj_final = self._compute_barrier_objective(self.vars)
                print(f"\n{'='*70}")
                print(f"  Amigo converged in {i} iterations")
                print(f"{'='*70}")
                print(f"  Objective value         {obj_final:>20.10e}")
                print(f"  NLP error               {overall_error:>20.10e}")
                print(f"  Primal infeasibility    {p_inf_nlp:>20.10e}")
                print(f"  Dual infeasibility      {d_inf_nlp:>20.10e}")
                print(f"  Complementarity         {c_inf_nlp:>20.10e}")
                print(f"  Barrier parameter       {self.barrier_param:>20.10e}")
                print(f"  Total iterations        {i:>20d}")
                print(f"{'='*70}")
            return CONVERGED, overall_error, acceptable_counter, precision_floor_count

        # Divergence check
        x_max = np.max(np.abs(self.vars.get_solution().get_array()))
        if x_max > diverging_iterates_tol:
            if comm_rank == 0:
                print(f"  Diverging iterates: max |x| = {x_max:.2e}")
            return DIVERGED, overall_error, acceptable_counter, precision_floor_count

        # Acceptable convergence
        is_acceptable = (
            overall_error <= acceptable_tol
            and d_inf_nlp <= acceptable_dual_inf_tol
            and p_inf_nlp <= acceptable_constr_viol_tol
            and c_inf_nlp <= acceptable_compl_inf_tol
        )
        if acceptable_iter > 0 and is_acceptable:
            acceptable_counter += 1
            if acceptable_counter >= acceptable_iter:
                if comm_rank == 0:
                    obj_acc = self._compute_barrier_objective(self.vars)
                    print(f"\n{'='*70}")
                    print(f"  Amigo converged to acceptable point in {i} iterations")
                    print(f"{'='*70}")
                    print(f"  Objective value         {obj_acc:>20.10e}")
                    print(f"  NLP error               {overall_error:>20.10e}")
                    print(f"  Primal infeasibility    {p_inf_nlp:>20.10e}")
                    print(f"  Dual infeasibility      {d_inf_nlp:>20.10e}")
                    print(f"  Complementarity         {c_inf_nlp:>20.10e}")
                    print(f"  Total iterations        {i:>20d}")
                    print(f"{'='*70}")
                return (
                    CONVERGED_ACCEPTABLE,
                    overall_error,
                    acceptable_counter,
                    precision_floor_count,
                )
        else:
            acceptable_counter = 0

        # Precision floor: bit-identical residuals
        rel_change = abs(res_norm - prev_res_norm) / max(res_norm, 1e-30)
        if rel_change < 1e-14 and i > 0:
            precision_floor_count += 1
        else:
            precision_floor_count = 0
        if precision_floor_count >= 3 and is_acceptable:
            if comm_rank == 0:
                print(
                    f"  Precision floor: residual unchanged "
                    f"for {precision_floor_count} iterations"
                )
            return (
                PRECISION_FLOOR,
                overall_error,
                acceptable_counter,
                precision_floor_count,
            )

        return CONTINUE, overall_error, acceptable_counter, precision_floor_count

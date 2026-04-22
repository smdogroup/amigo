"""Iteration-data assembly and progress-table printing.

Builds the per-iteration record (objective, NLP error, step sizes,
step norm, filter size, etc.) and prints the IPM progress table.
Expensive debug diagnostics live in newton_diagnostics.py.
"""

import sys
import numpy as np


class IterationLogger:
    """Iteration table and per-iteration data assembly."""

    def _build_iter_data(
        self,
        i,
        elapsed_time,
        res_norm,
        line_iters,
        alpha_x_prev,
        alpha_z_prev,
        x_index_prev,
        z_index_prev,
        inertia_corrector,
        theta_res,
        eta_res,
        filter_ls,
        outer_filter,
        options,
        mult_ind,
    ):
        """Assemble the dict of per-iteration diagnostics for logging/history."""
        iter_data = {
            "iteration": i,
            "time": elapsed_time,
            "residual": res_norm,
            "barrier_param": self.barrier_param,
            "line_iters": line_iters,
            "alpha_x": alpha_x_prev,
            "x_index": x_index_prev,
            "alpha_z": alpha_z_prev,
            "z_index": z_index_prev,
        }
        if inertia_corrector:
            iter_data.update(
                {
                    "theta": theta_res,
                    "eta": eta_res,
                    "inertia_delta": inertia_corrector.last_delta_w,
                }
            )
        if filter_ls:
            iter_data["filter_size"] = len(outer_filter)

        if options["barrier_strategy"] == "heuristic" and options["verbose_barrier"]:
            complementarity, xi = self.optimizer.compute_complementarity(self.vars)
            iter_data["xi"] = xi
            iter_data["complementarity"] = complementarity

        # NLP-level quantities (mu=0) for display
        d_inf_log, p_inf_log, c_inf_log = self.optimizer.compute_kkt_error_mu(
            0.0, self.vars, self.grad
        )
        s_d_log, s_c_log = self._compute_optimality_scaling(mult_ind)
        iter_data["inf_pr"] = p_inf_log
        iter_data["inf_du"] = d_inf_log
        iter_data["compl"] = c_inf_log
        iter_data["nlp_error"] = max(
            d_inf_log / s_d_log, p_inf_log, c_inf_log / s_c_log
        )
        iter_data["objective"] = self._compute_barrier_objective(self.vars)
        iter_data["step_norm"] = float(np.max(np.abs(self.px.get_array())))
        return iter_data

    def write_log(self, iteration, iter_data):
        """Print the IPM iteration table row for this iteration."""
        if iteration % 20 == 0:
            print(
                f"{'iter':>4s}  {'nlp_error':>9s}  {'objective':>12s}  "
                f"{'inf_pr':>9s}  {'inf_du':>9s}  {'compl':>9s}  "
                f"{'mu':>9s}  {'||d||':>9s}  {'delta_w':>8s}  "
                f"{'alpha_x':>8s}  {'alpha_z':>8s}  "
                f"{'ls':>2s}  {'filt':>4s}"
            )

        mu = iter_data.get("barrier_param", 1.0)
        delta_w = iter_data.get("inertia_delta", 0.0)
        nlp_err = iter_data.get("nlp_error", 0.0)
        obj = iter_data.get("objective", 0.0)
        inf_pr = iter_data.get("inf_pr", 0.0)
        inf_du = iter_data.get("inf_du", 0.0)
        compl = iter_data.get("compl", 0.0)
        step_norm = iter_data.get("step_norm", 0.0)
        ax = iter_data.get("alpha_x", 0.0)
        az = iter_data.get("alpha_z", 0.0)
        ls = iter_data.get("line_iters", 0)
        fsize = iter_data.get("filter_size", 0)

        dw_str = f"{delta_w:8.1e}" if delta_w > 0 else f"{'---':>8s}"

        print(
            f"{iteration:4d}  {nlp_err:9.2e}  {obj:12.5e}  "
            f"{inf_pr:9.2e}  {inf_du:9.2e}  {compl:9.2e}  "
            f"{mu:9.2e}  {step_norm:9.2e}  {dw_str}  "
            f"{ax:8.2e}  {az:8.2e}  "
            f"{ls:2d}  {fsize:4d}"
        )
        sys.stdout.flush()

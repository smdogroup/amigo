"""Expensive diagnostics for the Newton step, gated by check_update_step.

Reports the Newton quality ratio and quadratic-convergence indicator,
per-component KKT errors at the trial point, the top contributors to
the multiplier step, the largest bound-complementarity deviations
|z * gap - mu|, and the linear-solve accuracy ||K * px - rhs||.
"""

import numpy as np


class NewtonDiagnostics:
    """Expensive diagnostics used only when check_update_step is on."""

    def _print_newton_diagnostics(self, rhs_norm, res_norm_mu, mult_ind):
        """Print detailed Newton step diagnostics (check_update_step only).

        Evaluates the residual at the full Newton step to assess:
        - Newton quality ratio: ||F(x+dx)|| / ||F(x)|| (should be << 1)
        - Quadratic convergence indicator: ||F(x+dx)|| / ||F(x)||^2
        - Per-component KKT errors at the trial point
        """
        mu_changed = abs(self.barrier_param - res_norm_mu) > 1e-30

        # Evaluate residual at full Newton step (temporary)
        self.optimizer.apply_step_update(1.0, 1.0, self.vars, self.update, self.temp)
        self._update_gradient(self.temp.get_solution())
        trial_res = self.optimizer.compute_residual(
            self.barrier_param,
            self.temp,
            self.grad,
            self.res,
        )
        d_sq_t, p_sq_t, c_sq_t = self.optimizer.compute_kkt_error(self.temp, self.grad)
        # Restore gradient at current point
        self._update_gradient(self.vars.get_solution())

        nq_ratio = trial_res / max(rhs_norm, 1e-30)
        print(
            f"  Newton quality: ||F(x+dx)||={trial_res:.2e}, "
            f"||F_mu(x)||={rhs_norm:.2e}, "
            f"ratio={nq_ratio:.4f}, "
            f"quadratic={trial_res / max(rhs_norm**2, 1e-30):.2e}"
            f"{', mu_changed' if mu_changed else ''}"
        )
        print(
            f"  Trial KKT: dual={np.sqrt(d_sq_t):.2e}, "
            f"primal={np.sqrt(p_sq_t):.2e}, "
            f"comp={np.sqrt(c_sq_t):.2e}"
        )

        # Newton direction norms and step sizes
        dx_full = np.array(self.update.get_solution())
        dlam_vec = dx_full[mult_ind]
        dx_vec = dx_full[~mult_ind]
        print(
            f"  Newton: ||dx||={np.linalg.norm(dx_vec):.2e}, "
            f"||dlam||={np.linalg.norm(dlam_vec):.2e}"
        )

        # Top-5 multiplier step contributors
        dlam_abs = np.abs(dlam_vec)
        top_k = min(5, len(dlam_abs))
        top_idx = np.argsort(dlam_abs)[-top_k:][::-1]
        mult_positions = np.where(mult_ind)[0]
        x_sol = np.array(self.vars.get_solution())
        print(f"  dlam top-{top_k}: ", end="")
        for rank, li in enumerate(top_idx):
            gi = mult_positions[li]
            print(
                f"[{gi}]={dlam_vec[li]:+.4e}(lam={x_sol[gi]:+.4e})",
                end="  " if rank < top_k - 1 else "\n",
            )

        # Top-5 bound complementarity deviations |z*gap - mu|
        try:
            zl_arr = np.array(self.vars.get_zl())
            zu_arr = np.array(self.vars.get_zu())
            n_var = self.optimizer.get_num_design_variables()
            lbx = np.array(self.optimizer.get_lbx())
            ubx = np.array(self.optimizer.get_ubx())
            sl_arr = np.array(self.vars.get_sl())
            su_arr = np.array(self.vars.get_su())
            finite_l = np.isfinite(lbx)
            finite_u = np.isfinite(ubx)
            comp_l = np.where(finite_l, zl_arr * sl_arr, 0.0)
            comp_u = np.where(finite_u, zu_arr * su_arr, 0.0)
            mu = self.barrier_param
            dev_l = np.abs(comp_l - mu)
            dev_u = np.abs(comp_u - mu)
            all_dev = np.concatenate([dev_l, dev_u])
            all_comp = np.concatenate([comp_l, comp_u])
            top5 = np.argsort(all_dev)[-min(5, len(all_dev)) :][::-1]
            print(f"  bound comp top-5 |z*gap-mu|: ", end="")
            for rank, ci in enumerate(top5):
                side = "L" if ci < n_var else "U"
                vi = ci % n_var
                print(
                    f"x[{vi}]{side}={all_comp[ci]:.2e}(dev={all_dev[ci]:.2e})",
                    end="  " if rank < len(top5) - 1 else "\n",
                )
        except Exception as e:
            print(f"  (bound comp diagnostic failed: {e})")

    def _run_check_update_diagnostics(self, comm_rank):
        """Verify the Newton step and (optionally) the linear-solve accuracy."""
        hess = (self.mpi_problem if self.distribute else self.problem).create_matrix()
        self.optimizer.check_update(
            self.barrier_param,
            self.grad,
            self.vars,
            self.update,
            hess,
        )
        if comm_rank == 0 and hasattr(self.solver, "hess"):
            from scipy.sparse import csr_matrix as _sp_csr

            self.res.copy_device_to_host()
            self.px.copy_device_to_host()
            _rhs = self.res.get_array().copy()
            _px = self.px.get_array().copy()
            _data = self.solver.hess.get_data()
            _nr, _nc, _nnz, _rp, _cl = self.solver.hess.get_nonzero_structure()
            _K = _sp_csr((_data, _cl, _rp), shape=(_nr, _nc))
            _Kpx = _K @ _px
            _serr = np.linalg.norm(_Kpx - _rhs)
            _rnrm = np.linalg.norm(_rhs)

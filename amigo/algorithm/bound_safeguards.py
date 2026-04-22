"""Keep iterates strictly interior to the variable bounds.

Two complementary safeguards.  Slack flooring prevents the stored
slacks sl, su from collapsing to zero as mu shrinks, which would make
Sigma = Z/S ill-conditioned.  The adaptive tau rule caps the Newton
step by the fraction-to-boundary parameter tau = max(tau_min, 1 - mu),
so the iterate never touches a bound.
"""

import numpy as np


class BoundSafeguards:
    """Slack flooring and adaptive fraction-to-boundary tau."""

    def _ensure_positive_slacks(self, vars_obj, mu):
        """Floor stored slacks to prevent ill-conditioning at tiny mu.

        For each bounded primal where sl < s_min = eps * min(1, mu):
          new_sl = min(max(mu / z, s_min), slack_move * max(1, |bound|) + sl)
        Also adjusts the primal x to be consistent: x = lb + new_sl.
        """
        eps = np.finfo(np.float64).eps
        s_min = eps * min(1.0, mu)
        if s_min == 0.0:
            s_min = np.finfo(np.float64).tiny
        slack_move = 1.81898940354586e-12  # epsilon^{3/4}

        # TODO: move to backend: these array reads (bounds + slack/dual vectors) should be replaced by a single backend.floor_slacks(mu)
        sl = np.array(vars_obj.get_sl())
        su = np.array(vars_obj.get_su())
        lbx = np.array(self.optimizer.get_lbx())
        ubx = np.array(self.optimizer.get_ubx())
        zl = np.array(vars_obj.get_zl())
        zu = np.array(vars_obj.get_zu())
        fl = np.isfinite(lbx)
        fu = np.isfinite(ubx)

        adjusted = 0
        need_sync = False

        # Lower slacks
        if np.any(fl):
            sl_f = sl[fl]
            if np.min(sl_f) < s_min:
                small = sl_f < s_min
                idx = np.where(fl)[0][small]
                g = np.maximum(sl_f[small], 0.0)
                z_vals = np.maximum(zl[idx], np.finfo(np.float64).tiny)
                target = np.maximum(mu / z_vals, s_min)
                cap = slack_move * np.maximum(1.0, np.abs(lbx[idx])) + g
                sl[idx] = np.minimum(target, cap)
                adjusted += int(np.sum(small))
                need_sync = True

        # Upper slacks
        if np.any(fu):
            su_f = su[fu]
            if np.min(su_f) < s_min:
                small = su_f < s_min
                idx = np.where(fu)[0][small]
                g = np.maximum(su_f[small], 0.0)
                z_vals = np.maximum(zu[idx], np.finfo(np.float64).tiny)
                target = np.maximum(mu / z_vals, s_min)
                cap = slack_move * np.maximum(1.0, np.abs(ubx[idx])) + g
                su[idx] = np.minimum(target, cap)
                adjusted += int(np.sum(small))
                need_sync = True

        if need_sync:
            # Update stored slacks
            slacks_arr = vars_obj.get_slacks().get_array()
            n_p = len(lbx)
            slacks_arr[:n_p] = sl
            slacks_arr[n_p:] = su
            vars_obj.get_slacks().copy_host_to_device()

            # Keep primals consistent: x = lb + sl, x = ub - su
            sol_arr = vars_obj.get_solution().get_array()
            pi = np.where(~self._qf_mult_ind)[0]
            for i in np.where(fl)[0]:
                sol_arr[pi[i]] = lbx[i] + sl[i]
            for i in np.where(fu)[0]:
                sol_arr[pi[i]] = ubx[i] - su[i]
            vars_obj.get_solution().copy_host_to_device()

        return adjusted

    def _compute_adaptive_tau(self, barrier_param, tau_min=0.99):
        """Adaptive fraction-to-boundary: tau = max(tau_min, 1 - mu)."""
        return max(tau_min, 1.0 - barrier_param)

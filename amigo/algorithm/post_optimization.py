"""Model outputs and post-optimality sensitivities.

Evaluates the model's output vector at the converged iterate and
computes post-optimality derivatives through either the adjoint or
direct method.
"""

import numpy as np

from ..utils import tocsr


class PostOptimization:
    """Output evaluation and post-optimality sensitivities."""

    def compute_output(self):
        """Evaluate model outputs at the final iterate."""
        output = self.model.create_output_vector()
        x = self.vars.get_solution()
        self.problem.compute_output(x, output.get_vector())
        return output

    def compute_post_opt_derivatives(self, of=[], wrt=[], method="adjoint"):
        """Compute the post-optimality derivatives of the outputs.

        Parameters
        ----------
        of, wrt : list of str
            Output and input variable names.
        method : {"adjoint", "direct"}
            Use adjoint when len(of) < len(wrt), direct otherwise.
        """
        of_indices, of_map = self.model.get_indices_and_map(of)
        wrt_indices, wrt_map = self.model.get_indices_and_map(wrt)

        dfdx = np.zeros((len(of_indices), len(wrt_indices)))

        x = self.vars.get_solution()

        out_wrt_input = self.problem.create_output_jacobian_wrt_input()
        self.problem.output_jacobian_wrt_input(x, out_wrt_input)
        out_wrt_input = tocsr(out_wrt_input)

        out_wrt_data = self.problem.create_output_jacobian_wrt_data()
        self.problem.output_jacobian_wrt_data(x, out_wrt_data)
        out_wrt_data = tocsr(out_wrt_data)

        grad_wrt_data = self.problem.create_gradient_jacobian_wrt_data()
        self.problem.gradient_jacobian_wrt_data(x, grad_wrt_data)
        grad_wrt_data = tocsr(grad_wrt_data)

        self.optimizer.compute_diagonal(self.vars, self.diag)

        self.solver.factor(
            self._obj_scale,
            x,
            self.diag,
            post_hessian=self._hessian_scaling_fn,
        )

        if method == "adjoint":
            for i in range(len(of_indices)):
                idx = of_indices[i]
                self.res.get_array()[:] = -out_wrt_input[idx, :].toarray()
                self.solver.solve(self.res, self.px)
                adjx = grad_wrt_data.T @ self.px.get_array()
                dfdx[i, :] = out_wrt_data[idx, wrt_indices] + adjx[wrt_indices]
        elif method == "direct":
            grad_wrt_data = grad_wrt_data.tocsc()
            for i in range(len(wrt_indices)):
                idx = wrt_indices[i]
                self.res.get_array()[:] = -grad_wrt_data[:, idx].toarray().flatten()
                self.solver.solve(self.res, self.px)
                dirx = out_wrt_input @ self.px.get_array()
                dfdx[:, i] = out_wrt_data[of_indices, idx] + dirx[of_indices]

        return dfdx

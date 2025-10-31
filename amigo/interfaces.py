from __future__ import annotations
from typing import TYPE_CHECKING, Any
from .model import Model
import numpy as np


if TYPE_CHECKING:  # type-checkers see it; users don't need runtime package
    import openmdao.api as om


def _import_openmdao():
    try:
        import openmdao.api as om

        return om
    except Exception as exc:
        raise ImportError(
            "OpenMDAO integration requires the 'openmdao' extra. Install with: "
            "pip install 'amigo[openmdao]'"
        ) from exc


def ExternalOpenMDAOComponent(om_problem, am_model):
    om = _import_openmdao()

    class ExtOMComponent:
        def __init__(self, om_problem: om.Problem, am_model: Model):

            self.om_problem = om_problem
            self.am_model = am_model

            # Get the OpenMDAO names for the inputs
            self.obj = []
            self.cons = []
            self.dvs = []

            self.nvars = 0
            self.dv_mapping = {}
            self.ncon = 0
            self.con_mapping = {}

            for name in self.om_problem.model.get_objectives():
                self.obj.append(name)
            for name, meta in self.om_problem.model.get_constraints().items():
                self.cons.append(name)
                size = meta["size"]
                self.con_mapping[name] = np.arange(self.ncon, self.ncon + size)
                self.ncon += size
            for name, meta in self.om_problem.model.get_design_vars().items():
                self.dvs.append(name)
                size = meta["size"]
                self.dv_mapping[name] = np.arange(self.nvars, self.nvars + size)
                self.nvars += size

            self.rowp = np.arange(0, self.ncon * self.nvars + 1, self.nvars, dtype=int)
            self.cols = np.arange(0, self.ncon * self.nvars, dtype=int) % self.nvars

            self.jac_mapping = {}
            for con in self.cons:
                rows = self.con_mapping[con]
                for dv in self.dvs:
                    cols = self.dv_mapping[dv]

                    indices = np.zeros((len(rows), len(cols)))
                    for i in range(len(rows)):
                        indices[i, :] = rows[i] + np.arange(len(cols), dtype=int)

                    self.jac_mapping[(con, dv)] = indices.flatten()

            return

        def get_constraint_jacobian_csr(self):
            return self.ncon, self.nvars, self.rowp, self.cols

        def evaluate(self, x, con, grad, jac):
            # Set the design variables into the OpenMDAO model
            for name in self.dvs:
                self.om_problem.set_val(name, x[self.dv_mapping[name]])

            # Run the model
            self.om_problem.run_model()

            # Compute the objective and the objective gradient
            fobj = 0
            if len(self.obj) > 0:
                fobj = self.om_problem.get_val(self.obj[0])
                dfdx = self.om_problem.compute_totals(of=self.obj[0], wrt=self.dvs)

                for name in self.dvs:
                    grad[self.dv_mapping[name]] = dfdx[self.obj[0], name]

            # Extract the constraint values and constraint Jacobian
            if len(self.cons) > 0:
                for name in self.cons:
                    con[self.con_mapping[name]] = self.om_problem.get_val(name)
                dcdx = self.om_problem.compute_totals(of=self.cons, wrt=self.dvs)

                for of, wrt in dcdx:
                    jac[self.jac_mapping[(of, wrt)]] = dcdx[of, wrt]

            return fobj

    return ExtOMComponent(om_problem, am_model)


# def PostOptComponent(model, mapping):
#     om = _import_openmdao()

#     class PostOptExplicit(om.ExplicitComponent):
#         def initialize(self):
#             super().__init__(core_model)
#             self.options.declare("mapping", default=mapping, types=dict)

#         def setup(self):
#             m = self.options["mapping"]
#             # register OpenMDAO inputs/outputs from mapping
#             for name, spec in m.get("inputs", {}).items():
#                 self.add_input(name, **spec)
#             for name, spec in m.get("outputs", {}).items():
#                 self.add_output(name, **spec)

#         def compute(self, inputs, outputs):
#             # adapt inputs -> core, run, adapt outputs
#             x = {k: inputs[k] for k in self.options["mapping"].get("inputs", {})}
#             y = self.core.run(x)  # your pure function/class method
#             for k in self.options["mapping"].get("outputs", {}):
#                 outputs[k] = y[k]

#         def compute_partials(self, inputs, partials):
#             if hasattr(self.core, "jacobian"):
#                 J = self.core.jacobian(inputs)  # dict-of-dicts or similar
#                 for (of, wrt), val in J.items():
#                     partials[of, wrt] = val

from __future__ import annotations
from typing import TYPE_CHECKING
from .model import Model, ModelVector
from .optimizer import Optimizer
import numpy as np


if TYPE_CHECKING:
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

    class _ExternalOpenMDAOComponent:
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

    return _ExternalOpenMDAOComponent(om_problem, am_model)


def AmigoIndepVarComp(amigo_model, data_names):
    """
    Create an IndepVarComp that automatically uses defaults from Amigo metadata.

    Args:
        amigo_model: The Amigo Model instance
        data_names: List of Amigo data field names

    Returns:
        om.IndepVarComp with outputs set to Amigo metadata defaults

    """
    om = _import_openmdao()

    indeps = om.IndepVarComp()

    for name in data_names:
        # Get shape and default value from Amigo metadata
        indices = amigo_model.get_indices(name)
        meta = amigo_model.get_meta(name)
        val = np.full(indices.shape, meta["value"])

        # Correct name for OpenMDAO
        sanitized = name.replace(".", "_")
        indeps.add_output(sanitized, val=val)

    return indeps


def ExplicitOpenMDAOPostOptComponent(**kwargs):
    om = _import_openmdao()

    class _ExplicitOpenMDAOPostOptComponent(om.ExplicitComponent):
        def initialize(self):
            self.options.declare("data", types=list)
            self.options.declare("output", types=list)
            self.options.declare("data_mapping", types=dict)
            self.options.declare("output_mapping", types=dict)
            self.options.declare("model", types=Model)
            self.options.declare("x", types=ModelVector)
            self.options.declare("lower", types=ModelVector)
            self.options.declare("upper", types=ModelVector)
            self.options.declare("opt_options", default={}, types=dict)

        def setup(self):
            self.data = self.options["data"]
            self.output = self.options["output"]
            self.data_mapping = self.options["data_mapping"]
            self.output_mapping = self.options["output_mapping"]
            self.model = self.options["model"]
            self.x = self.options["x"]
            self.lower = self.options["lower"]
            self.upper = self.options["upper"]
            self.opt_options = self.options["opt_options"]

            self.opt = Optimizer(self.model, self.x, lower=self.lower, upper=self.upper)

            for name in self.data:
                indices = self.model.get_indices(name)
                meta = self.model.get_meta(name)
                om_name = self.data_mapping[name]
                # Use Amigo metadata default value
                default_val = np.full(indices.shape, meta["value"])
                self.add_input(om_name, shape=indices.shape, val=default_val)

            for name in self.output:
                indices = self.model.get_indices(name)
                om_name = self.output_mapping[name]
                # If a float is given when expecting vector, remove shape input
                if indices.shape == ():
                    self.add_output(om_name)
                else:
                    self.add_output(om_name, shape=indices.shape)

            self.declare_partials(of="*", wrt="*")
            return

        def compute(self, inputs, outputs):
            data = self.model.get_data_vector()
            for name in self.data:
                om_name = self.data_mapping[name]
                data[name] = inputs[om_name]
            self.opt.optimize(self.opt_options)

            out = self.opt.compute_output()
            for name in self.output:
                om_name = self.output_mapping[name]
                outputs[om_name] = out[name]
            # Cache inputs
            self._last_inputs = {
                name: inputs[self.data_mapping[name]].copy() for name in self.data
            }

        def compute_partials(self, inputs, partials):
            # Re-optimize if inputs changed since compute()
            inputs_changed = not hasattr(self, "_last_inputs")
            if not inputs_changed:
                for name in self.data:
                    om_name = self.data_mapping[name]
                    if not np.array_equal(inputs[om_name], self._last_inputs[name]):
                        inputs_changed = True
                        break

            # Also compute derivatives if they haven't been computed yet
            if inputs_changed or not hasattr(self, "of_map"):
                data = self.model.get_data_vector()
                for name in self.data:
                    om_name = self.data_mapping[name]
                    data[name] = inputs[om_name]
                self.opt.optimize(self.opt_options)
                self._last_inputs = {
                    name: inputs[self.data_mapping[name]].copy() for name in self.data
                }

                # Compute derivatives only when inputs change
                self.dfdx, self.of_map, self.wrt_map = (
                    self.opt.compute_post_opt_derivatives(of=self.output, wrt=self.data)
                )

            for of in self.of_map:
                open_of = self.output_mapping[of]
                for wrt in self.wrt_map:
                    open_wrt = self.data_mapping[wrt]
                    # Use np.ix_ for proper 2D submatrix extraction
                    of_indices = self.of_map[of]
                    wrt_indices = self.wrt_map[wrt]

                    # Handle both scalar and array indices
                    if np.isscalar(of_indices):
                        of_indices = [of_indices]
                    if np.isscalar(wrt_indices):
                        wrt_indices = [wrt_indices]

                    # Extract submatrix and reshape if needed
                    subjac = self.dfdx[np.ix_(of_indices, wrt_indices)]

                    # Flatten if both dimensions are arrays (for proper shape)
                    if len(of_indices) > 1 or len(wrt_indices) > 1:
                        partials[open_of, open_wrt] = subjac.flatten()
                    else:
                        partials[open_of, open_wrt] = subjac

    return _ExplicitOpenMDAOPostOptComponent(**kwargs)

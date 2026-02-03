"""
Robust optimal control of cart-pole under uncertainty using UQPCE + OpenMDAO + Amigo.

Architecture:
  Outer loop (OpenMDAO/SLSQP):
    - Design variable: u (num_time_steps+1 control values)
    - Objective: effort + beta * (pce_mean + gamma * sqrt(pce_variance))
    - UQPCE inside the optimization loop provides PCE mean & variance

  Inner loop (Amigo, per sample):
    - Fixed control u(t) and fixed (m2_k, L_k) from UQPCE sample matrix
    - Solves dynamics, returns cost = ||q(T) - q_target||^2
    - Derivatives via post-optimality adjoint

"""

from pathlib import Path
import numpy as np
import openmdao.api as om
import amigo as am

from uqpce import UQPCEGroup, interface


# Amigo components
class TrapezoidRule(am.Component):
    def __init__(self, final_time=2.0, num_time_steps=100):
        super().__init__()
        self.add_constant("dt", value=final_time / num_time_steps)
        self.add_input("q1")
        self.add_input("q2")
        self.add_input("q1dot")
        self.add_input("q2dot")
        self.add_constraint("res")

    def compute(self):
        dt = self.constants["dt"]
        q1 = self.inputs["q1"]
        q2 = self.inputs["q2"]
        q1dot = self.inputs["q1dot"]
        q2dot = self.inputs["q2dot"]
        self.constraints["res"] = q2 - q1 - 0.5 * dt * (q1dot + q2dot)


class CartComponent(am.Component):
    def __init__(self):
        super().__init__()
        self.add_constant("g", value=9.81)
        self.add_data("L", value=0.5)
        self.add_data("m1", value=1.0)
        self.add_data("m2", value=0.3)
        self.add_data("x", value=0.0)
        self.add_input("q", shape=(4), label="state")
        self.add_input("qdot", shape=(4), label="rate")
        self.add_constraint("res", shape=(4), label="residual")

    def compute(self):
        g = self.constants["g"]
        L0 = self.data["L"]
        m1 = self.data["m1"]
        m2 = self.data["m2"]
        x = self.data["x"]
        q = self.inputs["q"]
        qdot = self.inputs["qdot"]

        sint = am.sin(q[1])
        cost = am.cos(q[1])

        res = 4 * [None]
        res[0] = q[2] - qdot[0]
        res[1] = q[3] - qdot[1]
        res[2] = (m1 + m2 * sint * sint) * qdot[2] - (
            x + m2 * L0 * sint * q[3] * q[3] + m2 * g * cost * sint
        )
        res[3] = L0 * (m1 + m2 * sint * sint) * qdot[3] - (
            -x * cost - m2 * L0 * cost * sint * q[3] * q[3] - (m1 + m2) * g * sint
        )
        self.constraints["res"] = res


class Objective(am.Component):
    def __init__(self):
        super().__init__()
        self.add_constant("pi", value=np.pi)
        self.add_input("q", shape=(4))
        self.add_objective("obj")

    def compute(self):
        pi = self.constants["pi"]
        q = self.inputs["q"]
        self.objective["obj"] = (
            (q[0] - 2.0) ** 2 + (q[1] - pi) ** 2 + q[2] ** 2 + q[3] ** 2
        )


class CostOutput(am.Component):
    def __init__(self):
        super().__init__()
        self.add_constant("pi", value=np.pi)
        self.add_input("q0")
        self.add_input("q1")
        self.add_input("q2")
        self.add_input("q3")
        self.add_output("cost")

    def compute_output(self):
        pi = self.constants["pi"]
        q0 = self.inputs["q0"]
        q1 = self.inputs["q1"]
        q2 = self.inputs["q2"]
        q3 = self.inputs["q3"]
        self.outputs["cost"] = (q0 - 2.0) ** 2 + (q1 - pi) ** 2 + q2**2 + q3**2


class InitialConditions(am.Component):
    def __init__(self):
        super().__init__()
        self.add_input("q", shape=4)
        self.add_constraint("res", shape=4)

    def compute(self):
        q = self.inputs["q"]
        self.constraints["res"] = [q[0], q[1], q[2], q[3]]


def create_cart_model(module_name="cart_pole2", final_time=2.0, num_time_steps=100):
    cart = CartComponent()
    trap = TrapezoidRule(final_time=final_time, num_time_steps=num_time_steps)
    obj = Objective()
    ic = InitialConditions()
    cost_out = CostOutput()

    model = am.Model(module_name)
    model.add_component("cart", num_time_steps + 1, cart)
    model.add_component("trap", 4 * num_time_steps, trap)
    model.add_component("obj", 1, obj)
    model.add_component("ic", 1, ic)
    model.add_component("cost_out", 1, cost_out)

    for i in range(4):
        s, e = i * num_time_steps, (i + 1) * num_time_steps
        model.link(f"cart.q[:{num_time_steps}, {i}]", f"trap.q1[{s}:{e}]")
        model.link(f"cart.q[1:, {i}]", f"trap.q2[{s}:{e}]")
        model.link(f"cart.qdot[:-1, {i}]", f"trap.q1dot[{s}:{e}]")
        model.link(f"cart.qdot[1:, {i}]", f"trap.q2dot[{s}:{e}]")

    model.link("cart.q[0, :]", "ic.q[0, :]")
    model.link(f"cart.q[{num_time_steps}, :]", "obj.q[0, :]")
    model.link(f"cart.q[{num_time_steps}, 0]", "cost_out.q0[0]")
    model.link(f"cart.q[{num_time_steps}, 1]", "cost_out.q1[0]")
    model.link(f"cart.q[{num_time_steps}, 2]", "cost_out.q2[0]")
    model.link(f"cart.q[{num_time_steps}, 3]", "cost_out.q3[0]")

    model.link("cart.m1[1:]", "cart.m1[0]")
    model.link("cart.m2[1:]", "cart.m2[0]")
    model.link("cart.L[1:]", "cart.L[0]")

    return model


# Warm-start: solve nominal problem once
def solve_nominal(num_time_steps, final_time, u_profile):
    model = create_cart_model(num_time_steps=num_time_steps, final_time=final_time)
    model.initialize()

    data = model.get_data_vector()
    data["cart.L"] = 0.5
    data["cart.m1"] = 1.0
    data["cart.m2"] = 0.3
    for i in range(num_time_steps + 1):
        data[f"cart.x[{i}]"] = float(u_profile[i])

    x = model.create_vector()
    lower = model.create_vector()
    upper = model.create_vector()

    x["cart.q[:, 0]"] = np.linspace(0, 2.0, num_time_steps + 1)
    x["cart.q[:, 1]"] = np.linspace(0, np.pi, num_time_steps + 1)
    x["cart.q[:, 2]"] = 1.0
    x["cart.q[:, 3]"] = 1.0
    lower["cart.q"] = -float("inf")
    lower["cart.qdot"] = -float("inf")
    upper["cart.q"] = float("inf")
    upper["cart.qdot"] = float("inf")

    opt = am.Optimizer(model, x, lower=lower, upper=upper)
    opt.optimize(
        {
            "initial_barrier_param": 0.1,
            "convergence_tolerance": 5e-7,
            "max_line_search_iterations": 4,
            "max_iterations": 500,
        }
    )
    x.get_vector().copy_device_to_host()

    q_sol = np.zeros((num_time_steps + 1, 4))
    qdot_sol = np.zeros((num_time_steps + 1, 4))
    for i in range(4):
        q_sol[:, i] = x[f"cart.q[:, {i}]"]
        qdot_sol[:, i] = x[f"cart.qdot[:, {i}]"]

    cost = opt.compute_output()["cost_out.cost[0]"]
    print(f"Nominal solve converged. Cost: {cost:.6f}")
    return q_sol, qdot_sol


# OpenMDAO components
class AmigoCartPole(om.ExplicitComponent):
    """Amigo cart-pole dynamics solve for a single UQPCE sample."""

    def initialize(self):
        self.options.declare("num_time_steps", types=int, default=100)
        self.options.declare("final_time", types=float, default=2.0)
        self.options.declare("m2", types=float, default=0.3)
        self.options.declare("L", types=float, default=0.5)
        self.options.declare("warm_start_q", default=None)
        self.options.declare("warm_start_qdot", default=None)

    def setup(self):
        nts = self.options["num_time_steps"]
        self.add_input("u", shape=(nts + 1,))
        self.add_output("cost", val=0.0)
        self.declare_partials("cost", "u")

        self._model = create_cart_model(
            num_time_steps=nts, final_time=self.options["final_time"]
        )
        self._model.initialize()

        self._wrt_list = [f"cart.x[{i}]" for i in range(nts + 1)]

        self._opt = None
        self._converged = False
        self._last_q = self.options["warm_start_q"]
        self._last_qdot = self.options["warm_start_qdot"]

        self._opt_options = {
            "initial_barrier_param": 1.0,
            "convergence_tolerance": 1e-5,
            "max_line_search_iterations": 4,
            "max_iterations": 200,
        }

    def compute(self, inputs, outputs):
        nts = self.options["num_time_steps"]
        model = self._model

        data = model.get_data_vector()
        data["cart.L"] = self.options["L"]
        data["cart.m1"] = 1.0
        data["cart.m2"] = self.options["m2"]
        u = inputs["u"]
        for i in range(nts + 1):
            data[f"cart.x[{i}]"] = float(u[i])

        x = model.create_vector()
        lower = model.create_vector()
        upper = model.create_vector()

        if self._last_q is not None:
            for i in range(4):
                x[f"cart.q[:, {i}]"] = self._last_q[:, i]
                x[f"cart.qdot[:, {i}]"] = self._last_qdot[:, i]
        else:
            x["cart.q[:, 0]"] = np.linspace(0, 2.0, nts + 1)
            x["cart.q[:, 1]"] = np.linspace(0, np.pi, nts + 1)
            x["cart.q[:, 2]"] = 1.0
            x["cart.q[:, 3]"] = 1.0

        lower["cart.q"] = -float("inf")
        lower["cart.qdot"] = -float("inf")
        upper["cart.q"] = float("inf")
        upper["cart.qdot"] = float("inf")

        try:
            opt = am.Optimizer(model, x, lower=lower, upper=upper)
            opt.optimize(self._opt_options)
            x.get_vector().copy_device_to_host()

            q_sol = np.zeros((nts + 1, 4))
            qdot_sol = np.zeros((nts + 1, 4))
            for i in range(4):
                q_sol[:, i] = x[f"cart.q[:, {i}]"]
                qdot_sol[:, i] = x[f"cart.qdot[:, {i}]"]
            self._last_q = q_sol
            self._last_qdot = qdot_sol

            outputs["cost"] = opt.compute_output()["cost_out.cost[0]"]
            self._opt = opt
            self._converged = True
        except RuntimeError as e:
            print(
                f"  [Amigo m2={self.options['m2']:.3f} L={self.options['L']:.3f}] "
                f"Solver failed: {e}"
            )
            self._last_q = None
            self._last_qdot = None
            outputs["cost"] = 100.0
            self._opt = None
            self._converged = False

    def compute_partials(self, inputs, partials):
        nts = self.options["num_time_steps"]
        if not self._converged or self._opt is None:
            partials["cost", "u"] = np.zeros((1, nts + 1))
            return

        dfdx, _, wrt_map = self._opt.compute_post_opt_derivatives(
            of="cost_out.cost[0]", wrt=self._wrt_list, method="adjoint"
        )
        dcdu = np.array([dfdx[0, wrt_map[w]] for w in self._wrt_list])
        partials["cost", "u"] = dcdu.reshape(1, -1)


class ControlEffort(om.ExplicitComponent):
    """Trapezoidal integral of u^2."""

    def initialize(self):
        self.options.declare("num_time_steps", types=int, default=100)
        self.options.declare("final_time", types=float, default=2.0)

    def setup(self):
        nts = self.options["num_time_steps"]
        self.add_input("u", shape=(nts + 1,))
        self.add_output("effort", val=0.0)

        dt = self.options["final_time"] / nts
        w = np.full(nts + 1, dt)
        w[0] *= 0.5
        w[-1] *= 0.5
        self._w = w
        self.declare_partials("effort", "u")

    def compute(self, inputs, outputs):
        u = inputs["u"]
        outputs["effort"] = np.dot(self._w, u * u)

    def compute_partials(self, inputs, partials):
        partials["effort", "u"] = (2.0 * self._w * inputs["u"]).reshape(1, -1)


class PCERobustObjective(om.ExplicitComponent):
    """obj = pce_mean^2 + pce_variance  (UQPCE reference formulation)"""

    def setup(self):
        self.add_input("pce_mean")
        self.add_input("pce_variance")
        self.add_output("obj")
        self.declare_partials("obj", "pce_mean")
        self.declare_partials("obj", "pce_variance", val=1.0)

    def compute(self, inputs, outputs):
        mu = inputs["pce_mean"][0]
        var = inputs["pce_variance"][0]
        outputs["obj"] = mu**2 + var

    def compute_partials(self, inputs, partials):
        mu = inputs["pce_mean"][0]
        partials["obj", "pce_mean"] = 2.0 * mu
        partials["obj", "pce_variance"] = 1.0


# Main
def run(
    input_file,
    matrix_file,
    num_time_steps=100,
    final_time=2.0,
    beta=200.0,
    gamma=1.0,
    u_lower=-20.0,
    u_upper=20.0,
    max_iter=200,
):
    # UQPCE initialization
    (
        var_basis,
        norm_sq,
        resampled_var_basis,
        aleatory_cnt,
        epistemic_cnt,
        resp_cnt,
        order,
        variables,
        sig,
        run_matrix,
    ) = interface.initialize(input_file, matrix_file)

    nts = num_time_steps
    n_u = nts + 1
    print(f"UQPCE samples: {resp_cnt}")

    # Initial control profile (linear ramp)
    u_init = np.linspace(10.0, -10.0, n_u)

    # Warm-start with nominal parameters
    warm_q, warm_qdot = solve_nominal(nts, final_time, u_init)

    # OpenMDAO problem
    prob = om.Problem()
    mdl = prob.model

    # Independent variable: control profile u
    mdl.add_subsystem(
        "ivc",
        om.IndepVarComp("u", val=u_init),
        promotes_outputs=["u"],
    )

    # Control effort
    mdl.add_subsystem(
        "effort",
        ControlEffort(num_time_steps=nts, final_time=final_time),
        promotes_inputs=["u"],
        promotes_outputs=[("effort", "control_effort")],
    )

    # Parallel Amigo solves
    parallel = mdl.add_subsystem("parallel", om.ParallelGroup())
    for i in range(resp_cnt):
        parallel.add_subsystem(
            f"s{i}",
            AmigoCartPole(
                num_time_steps=nts,
                final_time=final_time,
                m2=float(run_matrix[i, 0]),
                L=float(run_matrix[i, 1]),
                warm_start_q=warm_q.copy(),
                warm_start_qdot=warm_qdot.copy(),
            ),
        )
        mdl.connect("u", f"parallel.s{i}.u")

    # Collect costs via MuxComp
    mux = mdl.add_subsystem("mux", om.MuxComp(vec_size=resp_cnt))
    mux.add_var("cost_samples", shape=(1,))
    for i in range(resp_cnt):
        mdl.connect(f"parallel.s{i}.cost", f"mux.cost_samples_{i}")

    # UQPCE inside the optimization loop
    sample_mean_est = 1.0
    mdl.add_subsystem(
        "UQPCE",
        UQPCEGroup(
            significance=sig,
            var_basis=var_basis,
            norm_sq=norm_sq,
            resampled_var_basis=resampled_var_basis,
            tail="both",
            epistemic_cnt=epistemic_cnt,
            aleatory_cnt=aleatory_cnt,
            uncert_list=["cost_samples"],
            tanh_omega=5e-2,
            sample_ref0=[0.0],
            sample_ref=[max(50.0, 2.0 * sample_mean_est + 1.0)],
        ),
    )
    mdl.connect("mux.cost_samples", "UQPCE.cost_samples")

    # Robust objective: pce_mean^2 + pce_variance
    mdl.add_subsystem("robust_obj", PCERobustObjective())
    mdl.connect("UQPCE.cost_samples:mean", "robust_obj.pce_mean")
    mdl.connect("UQPCE.cost_samples:variance", "robust_obj.pce_variance")

    # Optimizer
    prob.driver = om.pyOptSparseDriver(optimizer="IPOPT")
    prob.driver.options["print_results"] = True
    prob.driver.opt_settings["max_iter"] = max_iter
    prob.driver.opt_settings["tol"] = 1e-6
    prob.driver.opt_settings["print_level"] = 5
    prob.driver.opt_settings["hessian_approximation"] = "limited-memory"
    prob.driver.opt_settings["limited_memory_max_history"] = 15
    prob.driver.opt_settings["acceptable_tol"] = 1e-4
    prob.driver.opt_settings["acceptable_iter"] = 5

    mdl.add_design_var("u", lower=u_lower, upper=u_upper)
    mdl.add_objective("robust_obj.obj")
    mdl.add_constraint("control_effort", upper=100.0)

    prob.setup()
    prob.set_val("u", u_init)

    prob.run_driver()

    # Final results
    u_opt = prob.get_val("u")
    effort = prob.get_val("control_effort")[0]
    costs = prob.get_val("mux.cost_samples")
    pce_mean = prob.get_val("UQPCE.cost_samples:mean")[0]
    pce_var = prob.get_val("UQPCE.cost_samples:variance")[0]
    pce_std = np.sqrt(max(pce_var, 0.0))
    ci_upper = prob.get_val("UQPCE.cost_samples:ci_upper")[0]
    ci_lower = prob.get_val("UQPCE.cost_samples:ci_lower")[0]

    print(f"\n===== Final Results =====")
    print(f"Objective (pce_mean^2 + pce_variance): {pce_mean**2 + pce_var:.6f}")
    print(f"Control effort (constraint <= 100): {effort:.4f}")
    print(f"PCE mean terminal error: {pce_mean:.6f}")
    print(f"PCE variance:   {pce_var:.6f}")
    print(f"PCE std dev:    {pce_std:.6f}")
    print(f"95% CI:         [{ci_lower:.6f}, {ci_upper:.6f}]")
    print(f"Sample costs:   {costs.flatten()}")

    # --- Plot trajectories across UQPCE samples ---
    time = np.linspace(0, final_time, nts + 1)

    # Collect per-sample trajectories and parameters
    sample_data = []
    # Per-state terminal errors
    q_target = np.array([2.0, np.pi, 0.0, 0.0])
    state_names = ["pos (m)", "angle (rad)", "cart vel (m/s)", "ang vel (rad/s)"]
    print(f"\n--- Per-state terminal errors (q(T) - q_target) ---")
    print(f"{'Sample':>6}  {'m2':>6}  {'L':>6}  ", end="")
    print("  ".join(f"{name:>16}" for name in state_names))
    for i in range(resp_cnt):
        comp = getattr(prob.model.parallel, f"s{i}")
        q_final = comp._last_q[-1, :]
        errs = q_final - q_target
        print(f"{i:>6}  {run_matrix[i, 0]:>6.3f}  {run_matrix[i, 1]:>6.3f}  ", end="")
        print("  ".join(f"{e:>+16.6f}" for e in errs))

    # Plot control profile
    plot_control_profile(time, u_opt)

    return prob


def plot_control_profile(time, u_opt, filename="cart_pole_robust_control.png"):
    """Plot the optimal robust control profile."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(time, u_opt, color="#8242A2", linewidth=2.5)
    ax.set_xlabel("Time (s)", fontsize=13)
    ax.set_ylabel("Control force (N)", fontsize=13)
    ax.set_title("Optimal robust control profile", fontsize=15, fontweight="bold")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Saved control plot to {filename}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Robust cart-pole optimal control under uncertainty"
    )
    parser.add_argument("--input-file", type=str, default=None)
    parser.add_argument("--matrix-file", type=str, default=None)
    parser.add_argument("--num-time-steps", type=int, default=100)
    parser.add_argument("--final-time", type=float, default=2.0)
    parser.add_argument(
        "--beta", type=float, default=200.0, help="Penalty weight on robust cost"
    )
    parser.add_argument(
        "--gamma", type=float, default=1.0, help="Weight on std dev in robust cost"
    )
    parser.add_argument("--u-lower", type=float, default=-20.0)
    parser.add_argument("--u-upper", type=float, default=20.0)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--build", action="store_true", default=False)

    args = parser.parse_args()

    if args.build:
        model = create_cart_model(
            num_time_steps=args.num_time_steps, final_time=args.final_time
        )
        source_dir = Path(__file__).resolve().parent
        model.build_module(source_dir=source_dir)
        print("Amigo module built successfully.")

    if args.input_file is None or args.matrix_file is None:
        if not args.build:
            parser.error(
                "--input-file and --matrix-file are required unless --build is used alone"
            )
        import sys

        sys.exit(0)

    run(
        input_file=args.input_file,
        matrix_file=args.matrix_file,
        num_time_steps=args.num_time_steps,
        final_time=args.final_time,
        beta=args.beta,
        gamma=args.gamma,
        u_lower=args.u_lower,
        u_upper=args.u_upper,
        max_iter=args.max_iter,
    )

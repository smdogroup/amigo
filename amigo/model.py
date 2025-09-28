import numpy as np
import ast
import sys
import importlib
from .amigo import (
    OrderingType,
    reorder_model,
    VectorInt,
    OptimizationProblem,
    AliasTracker,
    NodeOwners,
    AMIGO_INCLUDE_PATH,
    A2D_INCLUDE_PATH,
    CSRMat,
)
from .component import Component
from scipy.sparse import spmatrix

try:
    from mpi4py.MPI import COMM_WORLD
except:
    COMM_WORLD = None

if sys.version_info < (3, 11):
    Self = object
    from typing import Union, List
else:
    from typing import Union, Self, List


def _import_class(module_name: str, class_name: str):
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def _parse_var_expr(expr: str):
    """
    Parse an expression like:
      - 'model.group.var'
      - 'model.group.var[:]'
      - 'model.group.var[:, 0]'

    path = ["model", "group", "var"]
    indices = None or tuple of slice/index

    Returns:
        path, indices
    """
    try:
        node = ast.parse(expr, mode="eval").body

        # Determine if it's an attribute or a subscript
        if isinstance(node, ast.Subscript):
            attr_node = node.value
            if sys.version_info < (3, 9):
                indices = _parse_indices_depr(node.slice)
            else:
                indices = _parse_indices(node.slice)
        elif isinstance(node, ast.Attribute):
            attr_node = node
            indices = None
        else:
            raise ValueError("Unsupported expression format.")

        # Extract the full attribute path
        path = _parse_attribute_path(attr_node)

        return path, indices
    except Exception as e:
        raise ValueError(f"Failed to parse expression '{expr}': {e}")


def _parse_attribute_path(attr):
    path = []
    while isinstance(attr, ast.Attribute):
        path.append(attr.attr)
        attr = attr.value
    if isinstance(attr, ast.Name):
        path.append(attr.id)
    else:
        raise ValueError("Invalid attribute chain")
    return list(reversed(path))


def _parse_indices(slice_node):
    if isinstance(slice_node, ast.Tuple):
        return tuple(_eval_ast_index(elt) for elt in slice_node.elts)
    else:
        return (_eval_ast_index(slice_node),)


def _eval_ast_index(node):
    if isinstance(node, ast.Slice):
        return slice(
            _eval_ast_index(node.lower) if node.lower else None,
            _eval_ast_index(node.upper) if node.upper else None,
            _eval_ast_index(node.step) if node.step else None,
        )
    elif isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.Name) and node.id == "None":
        return None
    else:
        return ast.literal_eval(node)


def _parse_indices_depr(slice_node):
    if sys.version_info < (3, 9):
        if isinstance(slice_node, ast.ExtSlice):
            return tuple(_eval_ast_index_depr(dim) for dim in slice_node.dims)
        elif isinstance(slice_node, ast.Tuple):
            return tuple(_eval_ast_index_depr(elt) for elt in slice_node.elts)
        elif isinstance(slice_node, ast.Index):
            return _eval_ast_index_depr(slice_node.value)
        else:
            return _eval_ast_index_depr(slice_node)
    else:
        raise NotImplementedError


def _eval_ast_index_depr(node):
    if sys.version_info < (3, 9):
        if isinstance(node, ast.Slice):
            return slice(
                _eval_ast_index_depr(node.lower) if node.lower else None,
                _eval_ast_index_depr(node.upper) if node.upper else None,
                _eval_ast_index_depr(node.step) if node.step else None,
            )
        elif isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Name) and node.id == "None":
            return None
        elif isinstance(node, ast.Index):
            return _eval_ast_index_depr(node.value)
        else:
            return ast.literal_eval(node)
    else:
        raise NotImplementedError


class GlobalIndexPool:
    def __init__(self):
        self.counter = 0

    def allocate(self, shape):
        size = np.prod(shape)
        indices = np.arange(self.counter, self.counter + size).reshape(shape)
        self.counter += size
        return indices


class ComponentGroup:
    def __init__(
        self,
        name: str,
        size: int,
        comp_obj: Component,
        var_shapes: dict,
        index_pool: GlobalIndexPool,
        data_shapes: dict,
        data_index_pool: GlobalIndexPool,
        out_shapes: dict,
        out_index_pool: GlobalIndexPool,
    ):
        self.name = name
        self.size = size
        self.comp_obj = comp_obj
        self.class_name = comp_obj.name

        # Set up the variables
        self.vars = {}
        for var_name, shape in var_shapes.items():
            self.vars[var_name] = index_pool.allocate(shape)

        # Set up the data
        self.data = {}
        for data_name, shape in data_shapes.items():
            self.data[data_name] = data_index_pool.allocate(shape)

        # Set up the outputs
        self.outputs = {}
        for out_name, shape in out_shapes.items():
            self.outputs[out_name] = out_index_pool.allocate(shape)

    def get_input_names(self):
        return self.comp_obj.get_input_names()

    def get_constraint_names(self):
        return self.comp_obj.get_constraint_names()

    def get_data_names(self):
        return self.comp_obj.get_data_names()

    def get_output_names(self):
        return self.comp_obj.get_output_names()

    def get_var(self, varname: str):
        return self.vars[varname]

    def get_data(self, name: str):
        return self.data[name]

    def get_meta(self, name):
        if name in self.comp_obj.inputs:
            return self.comp_obj.inputs.get_meta(name)
        elif name in self.comp_obj.constraints:
            return self.comp_obj.constraints.get_meta(name)
        elif name in self.comp_obj.data:
            return self.comp_obj.data.get_meta(name)
        elif name in self.comp_obj.objective:
            return self.comp_obj.objective.get_meta(name)
        elif name in self.comp_obj.constants:
            return self.comp_obj.constants.get_meta(name)
        elif name in self.comp_obj.outputs:
            return self.comp_obj.outputs.get_meta(name)
        else:
            raise ValueError(
                f"No input, constraint, data, objective or constant for {self.class_name}.{name}"
            )

    def get_indices(self, vars: dict):
        size = 0
        dim = 0
        for name in vars:
            shape = vars[name].shape
            size += np.prod(shape)

            if len(shape) == 1:
                dim += 1
            else:
                dim += np.prod(shape[1:])

        # Set the entries of the vectors
        array = np.zeros(size, dtype=int)

        if array.size > 0:
            array = array.reshape(-1, dim)

        offset = 0
        for name in vars:
            shape = vars[name].shape
            if len(shape) == 1:
                array[:, offset] = vars[name][:]
                offset += 1
            elif len(shape) == 2:
                for i in range(shape[1]):
                    array[:, offset] = vars[name][:, i]
                    offset += 1
            elif len(shape) == 3:
                for i in range(shape[1]):
                    for j in range(shape[2]):
                        array[:, offset] = vars[name][:, i, j]
                        offset += 1
        return array

    def create_group(self, module_name: str):
        if not self.comp_obj.is_compute_empty() or not self.comp_obj.is_output_empty():
            data_array = self.get_indices(self.data)
            vec_array = self.get_indices(self.vars)
            out_array = self.get_indices(self.outputs)

            data_vec = VectorInt(np.prod(data_array.shape))
            data_vec.get_array()[:] = data_array.ravel()
            var_vec = VectorInt(np.prod(vec_array.shape))
            var_vec.get_array()[:] = vec_array.ravel()
            out_vec = VectorInt(np.prod(out_array.shape))
            out_vec.get_array()[:] = out_array.ravel()

            # Create the object
            return _import_class(module_name, self.class_name)(
                self.size, data_vec, var_vec, out_vec
            )
        return None


class ModelVector:
    def __init__(self, model, x):
        self._model = model
        self._x = x

    def get_vector(self):
        return self._x

    def __getitem__(self, expr):
        if isinstance(expr, str):
            x_array = self._x.get_array()
            return x_array[self._model.get_indices(expr)]
        elif isinstance(expr, (int, np.integer, slice)):
            x_array = self._x.get_array()
            return x_array[expr]
        else:
            raise KeyError("Key type {expr} not accepted")

    def __setitem__(self, expr, item):
        if isinstance(expr, str):
            x_array = self._x.get_array()
            x_array[self._model.get_indices(expr)] = item
        elif isinstance(expr, (int, np.integer, slice)):
            x_array = self._x.get_array()
            x_array[expr] = item
        else:
            raise KeyError("Key type {expr} not accepted")


class Model:
    def __init__(self, module_name: str):
        """
        Initialize the model class.

        Args:
            module_name (str): Name of the module that contains the component classes
        """
        self.module_name = module_name
        self.comp = {}
        self.index_pool = GlobalIndexPool()
        self.data_index_pool = GlobalIndexPool()
        self.output_index_pool = GlobalIndexPool()
        self.links = []
        self._initialized = False

    def _get_group_shapes(self, size: int, var_shapes: dict):
        for var_name in var_shapes:
            if var_shapes[var_name] is None:
                var_shapes[var_name] = (size,)
            elif isinstance(var_shapes[var_name], tuple):
                var_shapes[var_name] = (size,) + var_shapes[var_name]
            else:
                var_shapes[var_name] = (size, var_shapes[var_name])
        return var_shapes

    def add_component(self, name: str, size: int, comp_obj: Component):
        """
        Add a component group to the model.

        This function adds the component group to the model. No ordering or linking
        operations are performed until initialize() is called. All inputs and constraints from
        comp_obj are referred to by: name.var or name.var[i, j], where numpy-type index
        slicing can be used to denote slices of variables.

        Args:
            name (str): Name of the component group
            size (int): Number of times that the component is repeated within the group
            comp_obj (Component): Class derived from a component object
        """
        if name in self.comp:
            raise ValueError(f"Cannot add two components with the same name")

        vs = comp_obj.get_var_shapes()
        var_shapes = self._get_group_shapes(size, vs)
        ds = comp_obj.get_data_shapes()
        data_shapes = self._get_group_shapes(size, ds)
        ts = comp_obj.get_output_shapes()
        out_shapes = self._get_group_shapes(size, ts)

        self.comp[name] = ComponentGroup(
            name,
            size,
            comp_obj,
            var_shapes,
            self.index_pool,
            data_shapes,
            self.data_index_pool,
            out_shapes,
            self.output_index_pool,
        )

        return

    def add_model(self, name: str, model: Self):
        """
        Add an entire model class as a sub-model.

        This function adds the entire sub-model class. The sub-model name must be unique,
        but the same model sub-class can be added more than once. All inputs and constraints
        from the sub-model are referred to by name.comp_name.var or name.comp_name.var[i, j],
        where numpy-type index slicing ca be used.

        Args:
            name (str): Name of the sub-model
            model (Model): An instance of the Model class object
        """
        if name in self.comp:
            raise ValueError(
                f"Cannot add a sub-model with the same name as a component"
            )

        # Add all of the sub-model components
        for comp_name in model.comp:
            sub_name = name + "." + comp_name
            sub_size = model.comp[comp_name].size
            sub_obj = model.comp[comp_name].comp_obj

            self.add_component(sub_name, sub_size, sub_obj)

        # Add all of the sub-model links (if any exist)
        for (
            src_expr,
            src_idx,
            tgt_expr,
            tgt_idx,
        ) in model.links:
            self.link(name + "." + src_expr, name + "." + tgt_expr, src_idx, tgt_idx)

        return

    def _get_slice_indices(self, all, slice, idx):
        if slice is None and idx is None:
            return all
        elif slice is None and idx is not None:
            return all[idx]
        elif slice is not None and idx is None:
            return all[slice]
        else:
            return all[slice][idx]

    def link(
        self,
        src_expr: str,
        tgt_expr: str,
        src_indices: Union[None, list, np.ndarray] = None,
        tgt_indices: Union[None, list, np.ndarray] = None,
    ):
        """
        Link two inputs, constraints, outputs or data components so that they are the same.

        You cannot link intputs to outputs or constraints . You can only link inputs to inputs
        and outputs to outputs, constraints to constraints and data to data. The inputs are the
        design variables. The data is constant data for each component.

        The inputs/constraints are specified as sub_model.group.var[0, 1] or sub_model.group.var[:, 1]
        or sub_model.group.var, or any sliced numpy view.

        The purpose of the link statements is to enforce which input variables are the same.
        If constraints or outputs are linked, then the resulting values are summed across
        all group components.

        Args:
            src_expr (str): Source variable name
            tgt_expr (str): Target variable name
            src_indices (list, np.ndarray): Optional source indices
            tgt_indices (list, np.ndarray): Optional target indices
        """

        # Should do some check here to see if the links are valid
        self.links.append((src_expr, src_indices, tgt_expr, tgt_indices))
        return

    def _init_indices(
        self,
        links: list,
        pool: GlobalIndexPool,
        vtype: str = "vars",
    ):
        # Allocate the AliasTracker
        tracker = AliasTracker(pool.counter)

        for a_expr, a_idx, b_expr, b_idx in links:
            a_path, a_slice = _parse_var_expr(a_expr)
            a_var = ".".join(a_path)

            b_path, b_slice = _parse_var_expr(b_expr)
            b_var = ".".join(b_path)

            # Make sure that the connection is the right type
            a_type = self._get_expr_type(a_var)
            b_type = self._get_expr_type(b_var)

            # Check if the types are consistent
            is_var = a_type == b_type and (a_type == "input" or b_type == "constraint")
            is_data = a_type == b_type and (a_type == "data")
            is_output = a_type == b_type and (a_type == "output")

            if (
                (vtype == "vars" and is_var)
                or (vtype == "data" and is_data)
                or (vtype == "output" and is_output)
            ):
                a_all = self.get_indices(a_var)
                a_indices = self._get_slice_indices(a_all, a_slice, a_idx)

                b_all = self.get_indices(b_var)
                b_indices = self._get_slice_indices(b_all, b_slice, b_idx)

                if a_indices.shape == b_indices.shape:
                    tracker.alias(a_indices.flatten(), b_indices.flatten())
                elif a_indices.size == 1:
                    a_temp = a_indices * np.ones(b_indices.shape, dtype=int)
                    tracker.alias(a_temp.flatten(), b_indices.flatten())
                elif b_indices.size == 1:
                    b_temp = b_indices * np.ones(a_indices.shape, dtype=int)
                    tracker.alias(a_indices.flatten(), b_temp.flatten())
                else:
                    raise ValueError(
                        f"Incompatible link {a_expr} {a_indices.shape} and {b_expr} {b_indices.shape}"
                    )
            elif (not is_var) and (not is_data):
                raise ValueError(
                    f"Cannot link {vtype} for {a_expr} {a_type} and {b_expr} {b_type}"
                )

        # Order the aliased variables first. These are
        counter, vars = tracker.assign_group_vars()

        # Order any remaining variables
        for name, comp in self.comp.items():
            if vtype == "vars":
                items = comp.vars.items()
            elif vtype == "output":
                items = comp.outputs.items()
            else:
                items = comp.data.items()

            for varname, array in items:
                # Set the variable indices
                arr = array.ravel()
                arr[:] = vars[arr]

        return counter

    def link_by_name(self, src_comp=None, tgt_comp=None, vtype="input"):
        """
        Link inputs, constraints, outputs or data with the same name between different components.

        Add links to the variables that are defined in different components but share a common name.
        This can be useful if the naming convention is consistent across the model. The variables must
        have a compatible shape.

        Args:
            src_comp (str): Source component name or None
            tgt_comp (str): Target component name or None
            vtype (str): Type of variable to link
        """

        if src_comp is None and tgt_comp is None:
            # Link everything
            for src in self.comp:
                for tgt in self.comp:
                    if src != tgt:
                        self.link_by_name(src_comp=src, tgt_comp=tgt, vtype=vtype)
        else:
            if vtype == "input":
                src_names = self.comp[src_comp].get_input_names()
                tgt_names = self.comp[tgt_comp].get_input_names()
            elif vtype == "constraint":
                src_names = self.comp[src_comp].get_constraint_names()
                tgt_names = self.comp[tgt_comp].get_constraint_names()
            elif vtype == "data":
                src_names = self.comp[src_comp].get_data_names()
                tgt_names = self.comp[tgt_comp].get_data_names()
            elif vtype == "output":
                src_names = self.comp[src_comp].get_output_names()
                tgt_names = self.comp[tgt_comp].get_output_names()

            for src in src_names:
                if src in tgt_names:
                    src_name = src_comp + "." + src
                    tgt_name = tgt_comp + "." + src
                    self.link(src_name, tgt_name)

        return

    def _reorder_indices(self, order_type, order_for_block=False):
        arrays = []
        for name, comp in self.comp.items():
            arrays.append(comp.get_indices(comp.vars))

        if order_for_block:
            constraint_indices = self._get_constraint_indices()
            iperm = reorder_model(
                order_type, arrays, constraint_indices=constraint_indices
            )
        else:
            iperm = reorder_model(order_type, arrays)

        # Apply the reordering to the variables
        for name, comp in self.comp.items():
            for varname, array in comp.vars.items():
                arr = array.ravel()
                arr[:] = iperm[arr]

        return

    def _get_constraint_indices(self):
        """
        Get the output indices. This must be called after _init_indices
        """
        # Get the indices of variable names
        temp = np.zeros(self.num_variables, dtype=int)
        for name, comp in self.comp.items():
            cons = comp.get_constraint_names()
            for conname in cons:
                temp[comp.vars[conname]] = 1

        return np.nonzero(temp)[0]

    def initialize(
        self, comm=COMM_WORLD, order_type=OrderingType.AMD, order_for_block=False
    ):
        """
        Initialize the variable indices for each component and resolve all links.
        """

        self.num_variables = self._init_indices(
            self.links, self.index_pool, vtype="vars"
        )
        self.data_size = self._init_indices(
            self.links, self.data_index_pool, vtype="data"
        )
        self.num_outputs = self._init_indices(
            self.links, self.output_index_pool, vtype="output"
        )

        self._reorder_indices(order_type, order_for_block)

        self.constraint_indices = self._get_constraint_indices()
        self.num_constraints = len(self.constraint_indices)

        self._initialized = True
        self.problem = self._create_opt_problem(comm=comm)

        return

    def _get_expr_type(self, name: str):
        path, indices = _parse_var_expr(name)
        comp_name = ".".join(path[:-1])
        name = path[-1]

        if name in self.comp[comp_name].get_input_names():
            return "input"
        elif name in self.comp[comp_name].get_constraint_names():
            return "constraint"
        elif name in self.comp[comp_name].get_data_names():
            return "data"
        elif name in self.comp[comp_name].get_output_names():
            return "output"
        else:
            raise ValueError(
                f"Name {comp_name}.{name} is neither an input, constraint, output or data"
            )

    def get_indices(self, name: str):
        """
        Get the indices associated with the variable.

        You can use this to access the indices of variables within the model. For instance,
        get_indices("sub_model.comp.vars") will return the indices for all of the vars
        under sub_model.comp and get_indices("sub_model.comp.vars[2:5, :]") will return
        a sliced version of the indices.

        Args:
            name (str): The name of the variable indices to retrieve
        """
        path, indices = _parse_var_expr(name)
        comp_name = ".".join(path[:-1])
        name = path[-1]

        if comp_name not in self.comp:
            raise ValueError(f"Component name {comp_name} not found")

        if name in self.comp[comp_name].vars:
            if indices is None:
                return self.comp[comp_name].get_var(name)
            else:
                return self.comp[comp_name].get_var(name)[indices]
        elif name in self.comp[comp_name].data:
            if indices is None:
                return self.comp[comp_name].get_data(name)
            else:
                return self.comp[comp_name].get_data(name)[indices]
        else:
            raise ValueError(
                f"Name {comp_name}.{name} is not an input, constraint, output or data name"
            )

    def get_meta(self, name: str):
        """
        Get meta data for the associated input, output, data, objective or constraint name
        """
        path, _ = _parse_var_expr(name)  # Ignore the indices here
        comp_name = ".".join(path[:-1])
        name = path[-1]

        if comp_name not in self.comp:
            raise ValueError(f"Component name {comp_name} not found")

        return self.comp[comp_name].get_meta(name)

    def _create_opt_problem(self, comm=COMM_WORLD):
        """
        Create the optimization problem object that is used to evaluate the gradient and
        Hessian of the Lagrangian.
        """

        if not self._initialized:
            raise RuntimeError(
                "Must call initialize before creating the optimization problem"
            )

        comm_size = 1
        if comm is not None:
            comm_size = comm.size

        objs = []
        # outs = []
        for name, comp in self.comp.items():
            obj = comp.create_group(self.module_name)
            if obj is not None:
                objs.append(obj)

        var_ranges = np.zeros(comm_size + 1, dtype=np.int32)
        var_ranges[1:] = self.num_variables
        var_owners = NodeOwners(comm, var_ranges)

        data_ranges = np.zeros(comm_size + 1, dtype=np.int32)
        data_ranges[1:] = self.data_size
        data_owners = NodeOwners(comm, data_ranges)

        output_ranges = np.zeros(comm_size + 1, dtype=np.int32)
        output_ranges[1:] = self.num_outputs
        output_owners = NodeOwners(comm, output_ranges)

        # Set the multipliers
        is_multiplier = VectorInt(self.num_variables)
        is_multiplier.get_array()[:] = 0
        is_multiplier.get_array()[self.constraint_indices] = 1
        prob = OptimizationProblem(
            comm, data_owners, var_owners, output_owners, is_multiplier, objs
        )

        return prob

    def get_data_vector(self):
        return ModelVector(self, self.problem.get_data_vector())

    def create_vector(self):
        return ModelVector(self, self.problem.create_vector())

    def create_output_vector(self):
        return ModelVector(self, self.problem.create_output_vector())

    def get_opt_problem(self):
        """Retrieve the optimization problem"""
        return self.problem

    def get_indices_from_list(self, names: List[str]):
        """
        Given a list of variable names, create a concatenated list of indices and a mapping between
        the names and indices.

        Args:
            names (list(str)): List of strings containing the names names

        Returns:
            indices (np.ndarray): Concatenated array of the indices
            idx_dict (dict): Name -> new index mapping
        """
        idx_list = []
        idx_dict = {}
        idx_count = 0
        for name in names:
            idx = self.get_indices(name).ravel()
            idx_list.append(idx)
            idx_dict[name] = np.arange(idx_count, idx_count + idx.size, dtype=int)
            idx_count += idx.size
        indices = np.concatenate(idx_list)

        return indices, idx_dict

    def extract_submatrix(
        self, A: Union[CSRMat, spmatrix], of: List[str], wrt: List[str]
    ):
        """
        Given the matrix A, find the sub-matrix A[indices[of], indices[wrt]]
        """

        of_indices, of_dict = self.get_indices_from_list(of)
        wrt_indices, wrt_dict = self.get_indices_from_list(wrt)

        if isinstance(A, spmatrix):
            Asub = A[of_indices, :][:, wrt_indices]
        elif isinstance(A, CSRMat):
            Asub = A.extract_submatrix(of_indices, wrt_indices)

        return Asub, of_dict, wrt_dict

    def get_names(self):
        """
        Get the scoped names of all inputs, constraints and data within the model
        """
        inputs = []
        cons = []
        data = []
        outputs = []

        for comp_name, comp in self.comp.items():
            for name in comp.get_input_names():
                inputs.append(".".join([comp_name, name]))
            for name in comp.get_constraint_names():
                cons.append(".".join([comp_name, name]))
            for name in comp.get_data_names():
                data.append(".".join([comp_name, name]))
            for name in comp.get_output_names():
                outputs.append(".".join([comp_name, name]))

        return inputs, cons, data, outputs

    def get_values_from_meta(self, meta_name: str):
        """
        Set values into the provided list or array.

        Note that this does not guarantee that the meta data is consistent between components.
        The last component added will overwrite whatever is set by other components without
        checking the consistency of the values.

        Args:
            meta_name (str) : The name of the meta data to place into the array

        Returns:
            x (ModelVector) : The meta values assigned to each component
        """

        if not self._initialized:
            raise RuntimeError(
                "Must call initialize before calling get_values_from_meta"
            )

        x = ModelVector(self, self.problem.create_vector())
        for comp_name, comp in self.comp.items():
            for var_name in comp.vars:
                name = comp_name + "." + var_name
                meta = self.get_meta(name)
                value = meta[meta_name]
                if value is None:
                    value = 0.0
                x[name] = value

        return x

    def build_module(
        self,
        comm=COMM_WORLD,
        compile_args=[],
        link_args=[],
        define_macros=[],
        debug=False,
    ):
        """
        Generate the model code and build it. Additional compile, link arguments and macros can be added here.
        """

        comm_rank = 0
        if comm is not None:
            comm_rank = comm.rank

        if comm_rank == 0:
            self._generate_cpp()
            self._build_module(
                compile_args=compile_args,
                link_args=link_args,
                define_macros=define_macros,
                debug=debug,
            )

        if comm is not None:
            comm.Barrier()

        return

    def _generate_cpp(self):
        """
        Generate the C++ header and pybind11 wrapper for the model.

        This code automatically generates the files module_name.h (containing the C++ Component
        definitions) and module_name.cpp (containing the pybind11 wrapper).

        The wrapper must be compiled before the optimization will run.
        """

        # C++ file contents
        cpp = '#include "a2dcore.h"\n'
        cpp += "namespace amigo {"

        # pybind11 file contents
        py11 = "#include <pybind11/numpy.h>\n"
        py11 += "#include <pybind11/pybind11.h>\n"
        py11 += "#include <pybind11/stl.h>\n"
        py11 += '#include "component_group.h"\n'
        py11 += f'#include "{self.module_name}.h"\n'
        py11 += "namespace py = pybind11;\n"

        mod_ident = "mod"
        py11 += f"PYBIND11_MODULE({self.module_name}, {mod_ident}) " + "{\n"

        # Write out the classes needed - class names must be unique
        # so we don't duplicate code
        class_names = {}
        for name in self.comp:
            class_name = self.comp[name].class_name
            if class_name not in class_names:
                compute_empty = self.comp[name].comp_obj.is_compute_empty()
                output_empty = self.comp[name].comp_obj.is_output_empty()
                if not compute_empty or not output_empty:
                    class_names[class_name] = True

                    # Generate the C++
                    cpp += self.comp[name].comp_obj.generate_cpp()

                # Generate the wrappers
                if not compute_empty or not output_empty:
                    py11 += (
                        self.comp[name].comp_obj.generate_pybind11(mod_ident=mod_ident)
                        + ";\n"
                    )

        cpp += "}\n"
        py11 += "}\n"

        filename = self.module_name + ".h"
        with open(filename, "w") as fp:
            fp.write(cpp)

        filename = self.module_name + ".cpp"
        with open(filename, "w") as fp:
            fp.write(py11)

        return

    def _build_module(
        self, compile_args=[], link_args=[], define_macros=[], debug=False
    ):
        """
        Quick setup for building the extension module. Some care is required with this.
        """
        from setuptools import setup, Extension
        from subprocess import check_output
        import pybind11
        import sys
        import os
        from pybind11.setup_helpers import Pybind11Extension, build_ext

        def get_mpi_flags():
            # Windows-specific MPI handling
            if sys.platform == "win32":
                # Microsoft MPI SDK paths
                mpi_sdk_base = r"C:\Program Files (x86)\Microsoft SDKs\MPI"
                inc_dirs = [os.path.join(mpi_sdk_base, "Include")]
                lib_dirs = [os.path.join(mpi_sdk_base, "Lib", "x64")]
                libs = ["msmpi"]  # Microsoft MPI library name
                return inc_dirs, lib_dirs, libs
            else:
                # Unix/Linux/Mac systems - use mpicxx
                # Split the output from the mpicxx command
                args = check_output(["mpicxx", "-show"]).decode("utf-8").split()

                # Determine whether the output is an include/link/lib command
                inc_dirs, lib_dirs, libs = [], [], []
                for flag in args:
                    if flag[:2] == "-I":
                        inc_dirs.append(flag[2:])
                    elif flag[:2] == "-L":
                        lib_dirs.append(flag[2:])
                    elif flag[:2] == "-l":
                        libs.append(flag[2:])

                return inc_dirs, lib_dirs, libs

        # Append the extra compile args list based on system type (allows for
        # compilaton on Windows vs. Linux/Mac)
        if sys.platform == "win32":
            compile_args += ["/std:c++17", "/permissive-"]
        else:
            compile_args += ["-std=c++17"]

        if debug:
            compile_args += ["-g", "-O0"]

        pybind11_include = pybind11.get_include()
        amigo_include = AMIGO_INCLUDE_PATH
        a2d_include = A2D_INCLUDE_PATH

        try:
            import mpi4py

            inc_dirs, lib_dirs, libs = get_mpi_flags()
            inc_dirs.append(mpi4py.get_include())
        except:
            inc_dirs, lib_dirs, libs = [], [], []

        # Add platform-specific libraries
        if sys.platform == "win32":
            openblas_root = r"C:\libs\openblas"
            inc_dirs += [os.path.join(openblas_root, "include")]
            lib_dirs += [os.path.join(openblas_root, "lib")]
            libs += ["openblas"]

        # Create the Extension
        all_inc_dirs = inc_dirs + [pybind11_include, amigo_include, a2d_include]
        ext_modules = [
            Extension(
                self.module_name,
                sources=[f"{self.module_name}.cpp"],
                depends=[f"{self.module_name}.h"],
                include_dirs=all_inc_dirs,
                libraries=libs,
                library_dirs=lib_dirs,
                extra_compile_args=compile_args,
                extra_link_args=link_args,
                define_macros=define_macros,
            )
        ]

        setup(
            name=f"{self.module_name}",
            ext_modules=ext_modules,
            script_args=["build_ext", "--inplace"],
            include_dirs=[amigo_include, pybind11_include, a2d_include],
        )

    def _build_tree_data(self, tree, name):
        subtree = {
            "name": name,
        }

        if name in self.comp:
            subtree["name"] = name
            subtree["class"] = self.comp[name].class_name
            subtree["size"] = self.comp[name].size

            data = {}
            for data_name in self.comp[name].get_data_names():
                data[data_name] = self.comp[name].get_meta(data_name).todict()
            subtree["data"] = data

            inputs = {}
            for input_name in self.comp[name].get_input_names():
                inputs[input_name] = self.comp[name].get_meta(input_name).todict()
            subtree["inputs"] = inputs

            cons = {}
            for con_name in self.comp[name].get_constraint_names():
                cons[con_name] = self.comp[name].get_meta(con_name).todict()
            subtree["constraints"] = cons
        else:
            children = []
            if "children" in tree:
                for child in tree["children"]:
                    children.append(
                        self._build_tree_data(tree["children"][child], child)
                    )
            subtree["children"] = children

        return subtree

    def _build_tree(self):
        tree = {"children": {}}
        for name in self.comp.keys():
            path = name.split(".")

            current = tree
            for index, part in enumerate(path):
                part_name = ".".join(path[: index + 1])
                if "children" not in current:
                    current["children"] = {}

                if part_name not in current["children"]:
                    current["children"][part_name] = {}
                current = current["children"][part_name]

        tree_data = {"module_name": self.module_name}
        children = []
        for child in tree["children"]:
            children.append(self._build_tree_data(tree["children"][child], child))
        tree_data["children"] = children

        return tree_data

    def get_serializable_data(self):

        # Create a model data dictionary
        model_data = self._build_tree()

        def _to_list(obj):
            if obj is None:
                return None
            elif isinstance(obj, (int, np.integer)):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, list):
                return obj  # Assuming it contains JSON-serializable elements
            else:
                raise TypeError(
                    f"Object {obj} of type {type(obj).__name__} is not JSON serializable"
                )

        # Serialize the link data
        links = []
        for src_expr, src_idx, tgt_expr, tgt_idx in self.links:
            spath, sslice = _parse_var_expr(src_expr)
            tpath, tslice = _parse_var_expr(tgt_expr)
            links.append(
                {
                    "src": (".".join(spath), str(sslice), _to_list(src_idx)),
                    "tgt": (".".join(tpath), str(tslice), _to_list(tgt_idx)),
                }
            )

        # Set the path names
        model_data["links"] = links

        return model_data

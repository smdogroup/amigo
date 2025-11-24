import numpy as np
import ast
import sys
import importlib
import subprocess
from pathlib import Path
from scipy.sparse import spmatrix
import pybind11
import networkx as nx
from .amigo import (
    OrderingType,
    MemoryLocation,
    reorder_model,
    VectorInt,
    OptimizationProblem,
    AliasTracker,
    NodeOwners,
    CSRMat,
    ExternalComponentGroup,
)
from .cmake_helper import get_cmake_dir
from .component import Component

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

    def get_output(self, name: str):
        return self.outputs[name]

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

    def create_group_object(self, module_name: str):
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


class ExternalComponent:
    def __init__(self, name, comp_obj, inputs=[], constraints=[]):
        self.name = name
        self.comp_obj = comp_obj
        self.inputs = inputs
        self.constraints = constraints
        return

    def get_input_names(self):
        return self.inputs

    def get_constraint_names(self):
        return self.constraints

    def create_group_object(self, input_idx, con_idx):
        nrows, ncols, jac_rowp, jac_cols = self.comp_obj.get_constraint_jacobian_csr()

        if nrows != len(con_idx):
            raise ValueError(
                "Number of constraint Jacobian rows inconsistent with the constraint dimensions"
            )
        if ncols != len(input_idx):
            raise ValueError(
                "Number of constraint Jacobian columns inconsistent with the input dimensions"
            )

        return ExternalComponentGroup(
            input_idx, con_idx, jac_rowp, jac_cols, self.comp_obj.evaluate
        )


class ModelVector:
    def __init__(self, model, x):
        self._model = model
        self._x = x

    def get_vector(self):
        return self._x

    def __getitem__(self, expr):
        if isinstance(expr, (str, list)):
            x_array = self._x.get_array()
            return x_array[self._model.get_indices(expr)]
        elif isinstance(expr, (int, np.integer, slice)):
            x_array = self._x.get_array()
            return x_array[expr]
        else:
            raise KeyError("Key type {expr} not accepted")

    def __setitem__(self, expr, item):
        if isinstance(expr, (str, list)):
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
        self.external_comp = {}
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

    def add_external_component(self, name: str, comp_obj, inputs=[], constraints=[]):
        """
        Add an external component to the model.

        Args:
            name (str): Name of the external component
            comp_obj (object): Python object
            inputs (List of str): List of strings of the input name
            constraints (List of str): List of strings of the constraint names
        """

        self.external_comp[name] = ExternalComponent(
            name, comp_obj, inputs, constraints
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

        # Add all of the sub-model external components
        for comp_name in model.external_comp:
            sub_name = name + "." + comp_name

            sub_inputs = []
            sub_constraints = []
            sub_group = model.external_comp[comp_name]
            sub_obj = sub_group.comp_obj

            for iname in enumerate(sub_group.inputs):
                sub_inputs.append(name + "." + iname)

            for iname in enumerate(sub_group.constraints):
                sub_constraints.append(name + "." + iname)

            self.add_external_component(sub_name, sub_obj, sub_inputs, sub_constraints)

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
            elif a_type != b_type:
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
            array = comp.get_indices(comp.vars)
            if array.size > 0:
                arrays.append(array)

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

    def get_indices(self, name: str | List[str]):
        """
        Get the indices associated with the variable.

        You can use this to access the indices of variables within the model. For instance,
        get_indices("sub_model.comp.vars") will return the indices for all of the vars
        under sub_model.comp and get_indices("sub_model.comp.vars[2:5, :]") will return
        a sliced version of the indices.

        Args:
            name (str or list of str): The name of the variable indices to retrieve

        Returns:
            indices (np.ndarray): Array of indices
        """

        if isinstance(name, list):
            idx_list = []
            for name_ in name:
                idx_list.append(self.get_indices(name_).ravel())
            if len(idx_list) > 0:
                indices = np.concatenate(idx_list)
            else:
                indices = []

            return indices
        else:
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
            elif name in self.comp[comp_name].outputs:
                if indices is None:
                    return self.comp[comp_name].get_output(name)
                else:
                    return self.comp[comp_name].get_output(name)[indices]
            else:
                raise ValueError(
                    f"Name {comp_name}.{name} is not an input, constraint, output or data name"
                )

    def get_indices_and_map(self, names: str | List[str]):
        """
        Given a list of variable names, create a concatenated list of indices and a mapping between
        the names and indices.

        Args:
            names (list(str)): String or list of strings containing the names

        Returns:
            indices (np.ndarray): Concatenated array of the indices
            idx_dict (dict): Name -> new index mapping
        """
        idx_list = []
        idx_dict = {}
        idx_count = 0

        if isinstance(names, list):
            for name in names:
                idx = self.get_indices(name).ravel()
                idx_list.append(idx)
                idx_dict[name] = np.arange(idx_count, idx_count + idx.size, dtype=int)
                idx_count += idx.size
        else:
            idx = self.get_indices(names).ravel()
            idx_list.append(idx)
            idx_dict[names] = np.arange(idx_count, idx_count + idx.size, dtype=int)
            idx_count += idx.size

        indices = np.concatenate(idx_list)

        return indices, idx_dict

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

        # Add the regular component groups
        for _, comp in self.comp.items():
            obj = comp.create_group_object(self.module_name)
            if obj is not None:
                objs.append(obj)

        # Add the external components
        for _, comp in self.external_comp.items():
            input_idx = self.get_indices(comp.inputs)
            con_idx = self.get_indices(comp.constraints)
            obj = comp.create_group_object(input_idx, con_idx)
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

    def create_data_vector(self):
        return ModelVector(self, self.problem.create_data_vector())

    def get_problem(self):
        """Retrieve the optimization problem"""
        return self.problem

    def extract_submatrix(
        self, A: Union[CSRMat, spmatrix], of: List[str], wrt: List[str]
    ):
        """
        Given the matrix A, find the sub-matrix A[indices[of], indices[wrt]]
        """

        of_indices, of_dict = self.get_indices_and_map(of)
        wrt_indices, wrt_dict = self.get_indices_and_map(wrt)

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

    def _guess_source_dir(self):
        """Make a guess for the source directory"""

        for name in self.comp:
            cls = self.comp[name].comp_obj
            module = sys.modules[cls.__module__]
            source = getattr(module, "__file__", None)

            if source is not None:
                return Path(source).resolve().parent

        return None

    def build_module(
        self,
        comm=COMM_WORLD,
        source_dir: str | Path | None = None,
        build_dir: str | Path | None = None,
    ):
        """
        Generate the model code and build it. Additional compile, link arguments and macros can be added here.
        """

        comm_rank = 0
        if comm is not None:
            comm_rank = comm.rank

        if comm_rank == 0:
            if source_dir is None:
                source_dir = self._guess_source_dir()

            self.generate_cpp()
            self._build_module(source_dir=source_dir, build_dir=build_dir)

        if comm is not None:
            comm.Barrier()

        return

    def generate_cpp(self):
        """
        Generate the C++ header and pybind11 wrapper for the model.

        This code automatically generates the files module_name.h (containing the C++ Component
        definitions) and module_name.cpp (containing the pybind11 wrapper).

        The wrapper must be compiled before the optimization will run.
        """

        # C++ file contents
        cpp = '#include "amigo.h"\n'
        cpp += '#include "a2dcore.h"\n'
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
        py11 += "#ifdef AMIGO_USE_OPENMP\n"
        py11 += "  constexpr amigo::ExecPolicy policy = amigo::ExecPolicy::OPENMP;\n"
        py11 += "#elif defined(AMIGO_USE_CUDA)\n"
        py11 += "  constexpr amigo::ExecPolicy policy = amigo::ExecPolicy::CUDA;\n"
        py11 += "#else\n"
        py11 += "  constexpr amigo::ExecPolicy policy = amigo::ExecPolicy::SERIAL;\n"
        py11 += "#endif\n"

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
        self, source_dir: str | Path, build_dir: str | Path = None, **kwargs
    ):
        """
        Build an extension module, utilizing the"""

        source_dir = Path(source_dir).resolve()
        if build_dir is None:
            build_dir = source_dir / "_amigo_build"

        build_dir = build_dir.resolve()
        build_dir.mkdir(parents=True, exist_ok=True)
        cmake_pybind11_dir = pybind11.get_cmake_dir()

        # Optionally drop a trivial CMakeLists.txt into the source_dir
        cmakelists = source_dir / "CMakeLists.txt"
        cmakelists.write_text(
            f"""\
cmake_minimum_required(VERSION 3.25)
project({self.module_name} LANGUAGES CXX)

find_package(amigo REQUIRED CONFIG)

if(AMIGO_ENABLE_CUDA)
    enable_language(CUDA) 
endif()

amigo_add_python_module(
    NAME {self.module_name}
    SOURCES {self.module_name}.cpp
)
"""
        )

        # Locate the installed Amigo CMake package inside the Python package
        amigo_cmake_dir = get_cmake_dir()
        print("amigo_cmake_dir = ", amigo_cmake_dir)

        # Cmake command
        cmake_cmd = [
            "cmake",
            "-S",
            str(source_dir),
            "-B",
            str(build_dir),
            f"-DCMAKE_PREFIX_PATH={amigo_cmake_dir}",
            f"-DPython3_EXECUTABLE={sys.executable}",
            f"-Dpybind11_DIR={cmake_pybind11_dir}",
        ]
        build_cmd = ["cmake", "--build", str(build_dir), "--config", "Release"]

        print("Running CMake commands from amigo")
        print(" ".join(cmake_cmd))
        print(" ".join(build_cmd))

        # Configure
        p = subprocess.Popen(
            cmake_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        for line in p.stdout:
            print(line, end="")

        p.wait()

        # Build
        p = subprocess.Popen(
            build_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        for line in p.stdout:
            print(line, end="")

        p.wait()

        return

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

    def create_graph(
        self,
        comp_list=None,
        timestep=None,
        var_shape="dot",
        comp_shape="square",
    ):
        """
        Create a networkx instance of the model structure
        """

        if not self._initialized:
            raise RuntimeError(
                "Must call initialize before creating the graph structure"
            )

        # Track aliases for each global variable ID and their types
        global_var_aliases = {}  # global_id -> list of (alias_string, vtype)
        comp_names = []

        if comp_list is None:
            comp_items = self.comp.items()
        else:
            comp_items = []
            for comp_name, comp in self.comp.items():
                if comp_name in comp_list:
                    comp_items.append((comp_name, comp))

        # Loop over all the components and add the various names for each variable
        # Also determine variable types from component metadata for robust coloring
        comp_time_indices = {}
        for comp_name, comp in comp_items:

            # Decide which timesteps to include and cache for edges
            if timestep is None:
                time_indices = range(comp.size)
            elif isinstance(timestep, int):
                time_indices = [timestep]
            elif isinstance(timestep, (tuple, range, list)):
                time_indices = timestep
            else:
                raise ValueError(
                    "Invalid timestep format. Use int, tuple, list, or range."
                )
            # Filter invalid indices now and cache
            filtered_time_indices = [i for i in time_indices if i < comp.size]
            comp_time_indices[comp_name] = filtered_time_indices

            # Prepare sets for quick type lookup
            input_names = set(comp.get_input_names())
            constraint_names = set(comp.get_constraint_names())

            for var_name, array in comp.vars.items():
                # Determine semantic type for this variable name
                if var_name in constraint_names:
                    vtype = "constraint"
                elif var_name in input_names:
                    vtype = "input"
                else:
                    vtype = None

                for i in filtered_time_indices:
                    for idx in np.ndindex(
                        array.shape[1:]
                    ):  # Loop over the position indices

                        index = array[(i,) + idx]
                        node_name = (
                            f"{comp_name}.{var_name}[{i}, {', '.join(map(str, idx))}]"
                        )

                        # Track aliases for this global variable ID
                        if index not in global_var_aliases:
                            global_var_aliases[index] = []
                        global_var_aliases[index].append((node_name, vtype))

            # Add the component names for only selected time steps:
            for i in filtered_time_indices:
                comp_names.append(f"{comp_name}[{i}]")

        # Set the variable names
        graph = nx.Graph()

        # Create separate nodes for each alias and track linkage relationships
        alias_to_node_id = {}  # alias_string -> node_id
        linkage_edges = []  # list of (node_id1, node_id2) for dashed edges

        # Add variable nodes (one per alias)
        node_counter = 0
        for global_id, aliases in global_var_aliases.items():
            if not aliases:
                continue

            # Create nodes for each alias
            alias_nodes = []
            for alias_name, vtype in aliases:
                node_id = node_counter
                node_counter += 1
                alias_to_node_id[alias_name] = node_id
                alias_nodes.append(node_id)

                # Determine color based on type
                node_color = (
                    "#decbe4"
                    if vtype == "constraint"
                    else ("#97c2fc" if vtype == "input" else "blue")
                )

                graph.add_node(
                    node_id,
                    label=alias_name,
                    title=alias_name,
                    shape=var_shape,
                    color=node_color,
                )

            # Add dashed linkage edges between aliases of the same global variable
            for i in range(len(alias_nodes)):
                for j in range(i + 1, len(alias_nodes)):
                    linkage_edges.append((alias_nodes[i], alias_nodes[j]))

        # Add all of the individual components within each component group
        # and build a mapping from (component name, timestep) -> node id
        comp_node_id = {}
        for i, name in enumerate(comp_names):
            node_id = node_counter + i
            graph.add_node(
                node_id, label=name, title=name, shape=comp_shape, color="#fed9a6"
            )

            # name has the form "group[i]"; parse back out (group, i)
            if "[" in name and name.endswith("]"):
                cname, idx_str = name.split("[")
                try:
                    i_val = int(idx_str[:-1])
                    comp_node_id[(cname, i_val)] = node_id
                except ValueError:
                    pass

        # Add edges between variables and components
        for comp_name, comp in comp_items:
            filtered_time_indices = comp_time_indices.get(comp_name, [])

            for var_name, array in comp.vars.items():
                for i in filtered_time_indices:
                    comp_index = comp_node_id.get((comp_name, i))
                    if comp_index is None:
                        continue
                    vars = array[i]
                    for idx in np.ndindex(vars.shape):
                        array_idx = (i,) + idx
                        global_id = array[array_idx]

                        # Find the alias node for this global variable
                        alias_name = (
                            f"{comp_name}.{var_name}[{i}, {', '.join(map(str, idx))}]"
                        )
                        if alias_name in alias_to_node_id:
                            var_node_id = alias_to_node_id[alias_name]
                            graph.add_edge(comp_index, var_node_id)

        # Add dashed linkage edges between aliased variables
        for node1, node2 in linkage_edges:
            graph.add_edge(node1, node2, dashes=True, color="gray")

        return graph

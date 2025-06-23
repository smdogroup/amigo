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
    AMIGO_INCLUDE_PATH,
    A2D_INCLUDE_PATH,
)
from .component import Component


if sys.version_info < (3, 9):
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
            return (_eval_ast_index_depr(slice_node.value),)
        else:
            return (_eval_ast_index_depr(slice_node),)
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

    def get_input_names(self):
        return self.comp_obj.get_input_names()

    def get_output_names(self):
        return self.comp_obj.get_output_names()

    def get_data_names(self):
        return self.comp_obj.get_data_names()

    def get_var(self, varname: str):
        return self.vars[varname]

    def get_data(self, name: str):
        return self.data[name]

    def get_meta(self, name):
        if name in self.comp_obj.inputs:
            return self.comp_obj.inputs.get_meta(name)
        elif name in self.comp_obj.outputs:
            return self.comp_obj.outputs.get_meta(name)
        elif name in self.comp_obj.data:
            return self.comp_obj.data.get_meta(name)
        elif name in self.comp_obj.objective:
            return self.comp_obj.objective.get_meta(name)
        elif name in self.comp_obj.constants:
            return self.comp_obj.constants.get_meta(name)
        else:
            raise ValueError(
                f"No input, output, data, objective or constant for {self.class_name}.{name}"
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

    def create_model(self, module_name: str):
        if not self.comp_obj.is_empty():
            data_array = self.get_indices(self.data)
            vec_array = self.get_indices(self.vars)

            data_vec = VectorInt(np.prod(data_array.shape))
            data_vec.get_array()[:] = data_array.ravel()
            vec = VectorInt(np.prod(vec_array.shape))
            vec.get_array()[:] = vec_array.ravel()

            # Create the object
            return _import_class(module_name, self.class_name)(data_vec, vec)
        return None


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
        self.links = []
        self._initialized = False

        self.input_names = {}
        self.output_names = {}
        self.data_names = {}

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
        operations are performed until initialize() is called. All inputs and outputs from
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

        self.comp[name] = ComponentGroup(
            name,
            size,
            comp_obj,
            var_shapes,
            self.index_pool,
            data_shapes,
            self.data_index_pool,
        )

        return

    def add_model(self, name: str, model: Self):
        """
        Add an entire model class as a sub-model.

        This function adds the entire sub-model class. The sub-model name must be unique,
        but the same model sub-class can be added more than once. All inputs and outputs
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
            self.link(name + "." + src_expr, src_idx, name + "." + tgt_expr, tgt_idx)

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
        Link two inputs, outputs or data components so that they are the same.

        You cannot link intputs to outputs. You can only link inputs to inputs and outputs
        to outputs and data to data. The outputs are used as constraints within the optimization problem.
        The inputs are the design variables. The data is constant data for each component.

        The inputs/outputs are specified as sub_model.group.var[0, 1] or sub_model.group.var[:, 1]
        or sub_model.group.var, or any sliced numpy view.

        The purpose of the link statements is to enforce which input variables are the same.
        If outputs are linked, then the resulting constraints are summed across all group components.

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
        type: str = "vars",
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
            is_var = a_type == b_type and (a_type == "input" or b_type == "output")
            is_data = a_type == b_type and a_type == "data"

            if (type == "vars" and is_var) or (type == "data" and is_data):
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
                    f"Cannot link {type} for {a_expr} {a_type} and {b_expr} {b_type}"
                )

        # Order the aliased variables first. These are
        counter, vars = tracker.assign_group_vars()

        # Order any remaining variables
        for name, comp in self.comp.items():
            if type == "vars":
                items = comp.vars.items()
            else:
                items = comp.data.items()

            for varname, array in items:
                # Set the variable indices
                arr = array.ravel()
                arr[:] = vars[arr]

        return counter

    def _reorder_indices(self, order_type, order_for_block=False):
        arrays = []
        for name, comp in self.comp.items():
            arrays.append(comp.get_indices(comp.vars))

        if order_for_block:
            output_indices = self._get_output_indices()
            iperm = reorder_model(order_type, arrays, output_indices=output_indices)
        else:
            iperm = reorder_model(order_type, arrays)

        # Apply the reordering to the variables
        for name, comp in self.comp.items():
            for varname, array in comp.vars.items():
                arr = array.ravel()
                arr[:] = iperm[arr]

        return

    def _get_output_indices(self):
        """
        Get the output indices. This must be called after _init_indices
        """
        # Get the indices of variable names
        temp = np.zeros(self.num_variables, dtype=int)
        for name, comp in self.comp.items():
            outputs = comp.get_output_names()
            for outname in outputs:
                temp[comp.vars[outname]] = 1

        return np.nonzero(temp)[0]

    def initialize(self, order_type=OrderingType.AMD, order_for_block=False):
        """
        Initialize the variable indices for each component and resolve all links.
        """

        self.num_variables = self._init_indices(
            self.links, self.index_pool, type="vars"
        )
        self.data_size = self._init_indices(
            self.links, self.data_index_pool, type="data"
        )

        self._reorder_indices(order_type, order_for_block)

        self.output_indices = self._get_output_indices()
        self.num_constraints = len(self.output_indices)

        self._initialized = True
        self.problem = self._create_opt_problem()

        return

    def _get_expr_type(self, name: str):
        path, indices = _parse_var_expr(name)
        comp_name = ".".join(path[:-1])
        name = path[-1]

        if name in self.comp[comp_name].get_input_names():
            return "input"
        elif name in self.comp[comp_name].get_output_names():
            return "output"
        elif name in self.comp[comp_name].get_data_names():
            return "data"
        else:
            raise ValueError(
                f"Name {comp_name}.{name} is neither an input, output or data"
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
                f"Name {comp_name}.{name} is not an input, output or data name"
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

    def _create_opt_problem(self):
        """
        Create the optimization problem object that is used to evaluate the gradient and
        Hessian of the Lagrangian.
        """

        if not self._initialized:
            raise RuntimeError(
                "Must call initialize before creating the optimization problem"
            )

        objs = []
        for name, comp in self.comp.items():
            obj = comp.create_model(self.module_name)
            if obj is not None:
                objs.append(obj)

        return OptimizationProblem(
            self.data_size,
            self.num_variables,
            self.output_indices,
            objs,
        )

    def get_opt_problem(self):
        """Retrieve the optimization problem"""
        return self.problem

    def get_values_from_meta(
        self, meta_name: str, x: Union[None, List, np.ndarray] = None
    ):
        """
        Set values into the provided list or array.

        Note that this does not guarantee that the meta data is consistent between components.
        The last component added will overwrite whatever is set by other components without
        checking the consistency of the values.

        Args:
            meta_name (str) : The name of the meta data to place into the array

        Returns:
            x (np.ndarray) : The meta values assigned to each component
        """

        if not self._initialized:
            raise RuntimeError(
                "Must call initialize before calling get_values_from_meta"
            )

        if x is None:
            x = np.zeros(self.num_variables)
        for comp_name, comp in self.comp.items():
            for var_name in comp.vars:
                name = comp_name + "." + var_name
                meta = self.get_meta(name)
                value = meta[meta_name]
                if value is None:
                    value = 0.0
                x[self.get_indices(name)] = value

        return x

    def build_module(self, compile_args=[], link_args=[], define_macros=[]):
        """
        Generate the model code and build it. Additional compile, link arguments and macros can be added here.
        """

        self._generate_cpp()
        self._build_module(
            compile_args=compile_args, link_args=link_args, define_macros=define_macros
        )

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
                if not self.comp[name].comp_obj.is_empty():
                    class_names[class_name] = True

                    # Generate the C++
                    cpp += self.comp[name].comp_obj.generate_cpp()

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

    def _build_module(self, compile_args=[], link_args=[], define_macros=[]):
        """
        Quick setup for building the extension module. Some care is required with this.
        """
        from setuptools import setup, Extension
        import pybind11
        import sys
        from pybind11.setup_helpers import Pybind11Extension, build_ext

        # Append the extra compile args list based on system type (allows for
        # compilaton on Windows vs. Linux/Mac)
        if sys.platform == "win32":
            compile_args += ["/std:c++17", "/permissive-"]
        else:
            compile_args += ["-std=c++17"]

        pybind11_include = pybind11.get_include()
        amigo_include = AMIGO_INCLUDE_PATH
        a2d_include = A2D_INCLUDE_PATH

        # Create the Extension
        ext_modules = [
            Extension(
                self.module_name,
                sources=[f"{self.module_name}.cpp"],
                depends=[f"{self.module_name}.h"],
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

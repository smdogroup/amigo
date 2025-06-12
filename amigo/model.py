import numpy as np
import ast
import sys
import importlib
from .amigo import VectorInt, OptimizationProblem, AMIGO_INCLUDE_PATH, A2D_INCLUDE_PATH
from .component import Component
from typing import Self
from collections import defaultdict


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
            return _eval_ast_index(node.value)
        else:
            return ast.literal_eval(node)
    else:
        raise NotImplementedError


class AliasTracker:
    def __init__(self, size: int):
        self.parent = np.arange(size, dtype=int)
        self.rank = np.zeros(size, dtype=int)

    def _find(self, var: int):
        if self.parent[var] != var:
            self.parent[var] = self._find(self.parent[var])
        return self.parent[var]

    def alias(self, var1: int, var2: int):
        root1 = self._find(var1)
        root2 = self._find(var2)
        if root1 == root2:
            return
        # Union by rank
        if self.rank[root1] < self.rank[root2]:
            self.parent[root1] = root2
        else:
            self.parent[root2] = root1
            if self.rank[root1] == self.rank[root2]:
                self.rank[root1] += 1

    def get_alias_group(self, var: int):
        root = self._find(var)
        return [v for v in range(len(self.parent)) if self._find(v) == root]

    def all_groups(self):
        groups = defaultdict(list)
        for var in range(len(self.parent)):
            root = self._find(var)
            groups[root].append(var)
        return list(groups.values())


class GlobalIndexPool:
    def __init__(self):
        self.counter = 0

    def allocate(self, shape):
        size = np.prod(shape)
        indices = np.arange(self.counter, self.counter + size).reshape(shape)
        self.counter += size
        return indices


class ComponentGroup:
    def __init__(self, name: str, size: int, comp_obj, var_shapes, index_pool):
        self.name = name
        self.size = size
        self.comp_obj = comp_obj
        self.class_name = comp_obj.name
        self.vars = {}
        for var_name, shape in var_shapes.items():
            self.vars[var_name] = index_pool.allocate(shape)

    def get_var(self, varname):
        return self.vars[varname]

    def create_model(self, module_name: str):
        size = 0
        dim = 0
        for name in self.vars:
            shape = self.vars[name].shape
            size += np.prod(shape)

            if len(shape) == 1:
                dim += 1
            else:
                dim += np.prod(shape[1:])

        # Set the entries of the vectors
        vec = VectorInt(size)
        array = vec.get_array()

        offset = 0
        for name in self.vars:
            shape = self.vars[name].shape
            if len(shape) == 1:
                array[offset::dim] = self.vars[name][:]
                offset += 1
            elif len(shape) == 2:
                for i in range(shape[1]):
                    array[offset::dim] = self.vars[name][:, i]
                    offset += 1
            elif len(shape) == 3:
                for i in range(shape[1]):
                    for i in range(shape[2]):
                        array[offset::dim] = self.vars[name][:, i, j]
                        offset += 1

        # Create the object
        return _import_class(module_name, self.class_name)(vec)


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
        self.links = []
        self._initialized = False

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

        var_shapes = comp_obj.get_var_shapes()

        for var_name in var_shapes:
            if var_shapes[var_name] is None:
                var_shapes[var_name] = (size,)
            elif isinstance(var_shapes[var_name], tuple):
                var_shapes[var_name] = (size,) + var_shapes[var_name]
            else:
                var_shapes[var_name] = (size, var_shapes[var_name])

        self.comp[name] = ComponentGroup(
            name, size, comp_obj, var_shapes, self.index_pool
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
        for src_expr, tgt_expr in model.links:
            self.link(name + "." + src_expr, name + "." + tgt_expr)

        return

    def link(self, src_expr: str, tgt_expr: str):
        """
        Link two variables so that they are the same.

        You cannot link intputs to outputs. You can only link inputs to inputs and outputs 
        to outputs. The outputs are used as constraints within the optimization problem.
        The inputs are the design variables.

        The variables are specified as sub_model.group.var[0, 1] or sub_model.group.var[:, 1]
        or sub_model.group.var, or any sliced numpy view.

        The purpose of the link statements is to enforce which input variables are the same.
        If outputs are linked, then the resulting constraints are summed across all group components.

        Args:
            src_expr (str): Source variable name
            tgt_expr (str): Target variable name
        """
        self.links.append((src_expr, tgt_expr))
        return

    def initialize(self):
        """
        Initialize the variable indices for each component and resolve all links.
        """

        # Allocate the AliasTracker
        tracker = AliasTracker(self.index_pool.counter)

        for a_expr, b_expr in self.links:
            a_path, a_slice = _parse_var_expr(a_expr)
            a_var = ".".join(a_path)
            a_indices = self.get_var_indices(a_var)[a_slice]

            b_path, b_slice = _parse_var_expr(b_expr)
            b_var = ".".join(b_path)
            b_indices = self.get_var_indices(b_var)[b_slice]

            if a_indices.shape != b_indices.shape:
                raise ValueError(
                    f"Incompatible shapes {a_expr} {a_indices.shape} and {b_expr} {b_indices.shape}"
                )

            for a, b in zip(a_indices.flatten(), b_indices.flatten()):
                tracker.alias(a, b)

        # Set up the variables so that they are contiguous
        vars = -np.ones(self.index_pool.counter, dtype=int)

        # Order the variables
        counter = 0

        # Order any aliased variables
        groups = tracker.all_groups()
        for group in groups:
            for index in group:
                vars[index] = counter

            counter += 1

        # Order any remaining variables
        for name, comp in self.comp.items():
            for varname, array in comp.vars.items():
                arr = array.ravel()

                for i in range(arr.shape[0]):
                    if vars[arr[i]] == -1:
                        vars[arr[i]] = counter
                        counter += 1

                    arr[i] = vars[arr[i]]

        self.num_variables = counter
        self._initialized = True

        return

    def get_var_indices(self, name: str):
        """
        Get the indices associated with the variable.

        You can use this to access the indices of variables within the model. For instance,
        get_var_inidces("sub_model.comp.vars") will return the indices for all of the vars
        under sub_model.comp and get_var_inidces("sub_model.comp.vars[2:5, :]") will return
        a sliced version of the indices.

        Args:
            name (str): The name of the variable indices to retrieve
        """
        path, indices = _parse_var_expr(name)
        comp_name = ".".join(path[:-1])
        var_name = path[-1]

        if indices is None:
            return self.comp[comp_name].get_var(var_name)
        else:
            return self.comp[comp_name].get_var(var_name)[indices]

    def create_opt_problem(self):
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
            objs.append(comp.create_model(self.module_name))

        return OptimizationProblem(self.num_variables, objs)

    def generate_cpp(self):
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

    def build_module(self):
        """
        Quick setup for building the extension module. Some care is required with this.
        """
        from setuptools import setup, Extension
        import pybind11
        from pybind11.setup_helpers import Pybind11Extension, build_ext

        pybind11_include = pybind11.get_include()
        amigo_include = AMIGO_INCLUDE_PATH
        a2d_include = A2D_INCLUDE_PATH

        # Create the Extension
        ext_modules = [
            Extension(
                self.module_name,
                sources=[f"{self.module_name}.cpp"],
                depends=[f"{self.module_name}.h"],
                extra_compile_args=["-std=c++17"],
            )
        ]

        setup(
            name=f"{self.module_name}",
            ext_modules=ext_modules,
            script_args=["build_ext", "--inplace"],
            include_dirs=[amigo_include, pybind11_include, a2d_include],
        )

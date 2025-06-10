import numpy as np
import ast
import importlib
from .amigo import VectorInt, OptimizationProblem


def import_class(module_name, class_name):
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


class GlobalIndexPool:
    def __init__(self):
        self.counter = 0

    def allocate(self, shape):
        size = np.prod(shape)
        indices = np.arange(self.counter, self.counter + size).reshape(shape)
        self.counter += size
        return indices


class ComponentSet:
    def __init__(self, name, comp_obj, var_shapes, index_pool):
        self.name = name
        self.comp_obj = comp_obj
        self.class_name = comp_obj.name
        self.vars = {}
        for var_name, shape in var_shapes.items():
            self.vars[var_name] = index_pool.allocate(shape)

    def get_var(self, varname):
        return self.vars[varname]
    
    def create_model(self, module_name):
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
        return import_class(module_name, self.class_name)(vec)


class Model:
    def __init__(self, module_name):
        self.module_name = module_name
        self.comp = {}
        self.index_pool = GlobalIndexPool()
        self.connections = []

    def generate_cpp(self):

        cpp = '#include "a2dcore.h"\n'
        cpp += "namespace amigo {"

        for name in self.comp:
            cpp += self.comp[name].comp_obj.generate_cpp()

        cpp += "}"

        filename = self.module_name + ".h"
        with open(filename, "w") as fp:
            fp.write(cpp)

        py11 = "#include <pybind11/numpy.h>\n"
        py11 += "#include <pybind11/pybind11.h>\n"
        py11 += "#include <pybind11/stl.h>\n"
        py11 += '#include "serial_component_set.h"\n'
        py11 += f'#include "{self.module_name}.h"\n'
        py11 += "namespace py = pybind11;\n"

        mod_ident = "mod"
        py11 += f"PYBIND11_MODULE({self.module_name}, {mod_ident}) " + "{\n"

        for name in self.comp:
            py11 += (
                self.comp[name].comp_obj.generate_pybind11(mod_ident=mod_ident) + ";\n"
            )

        py11 += "}\n"

        filename = self.module_name + ".cpp"
        with open(filename, "w") as fp:
            fp.write(py11)

        return

    def add(self, name, size, comp_obj):
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

        self.comp[name] = ComponentSet(name, comp_obj, var_shapes, self.index_pool)

        return

    def connect(self, src_expr: str, tgt_expr: str):
        src_comp, src_var, src_indices = self._parse_expr(src_expr)
        tgt_comp, tgt_var, tgt_indices = self._parse_expr(tgt_expr)

        src_array = self.comp[src_comp].get_var(src_var)
        tgt_array = self.comp[tgt_comp].get_var(tgt_var)

        src_view = src_array[src_indices]
        tgt_view = tgt_array[tgt_indices]

        if src_view.shape != tgt_view.shape:
            raise ValueError(
                f"Shape mismatch: {src_expr} {src_view.shape} vs {tgt_expr} {tgt_view.shape}"
            )

        # Connect by replacing target values with source view
        tgt_array[tgt_indices] = src_view

        self.connections.append((src_expr, tgt_expr))

    def _parse_expr(self, expr):
        """
        Parse a variable slice expression like 'comp.var[i,j]' or 'comp.var[:, 0]'
        """
        try:
            node = ast.parse(expr, mode="eval").body
            if not isinstance(node, ast.Subscript):
                raise ValueError(
                    "Expression must include subscript, like 'comp.var[i,j]'."
                )

            attr = node.value
            if not isinstance(attr, ast.Attribute) or not isinstance(
                attr.value, ast.Name
            ):
                raise ValueError("Expected format 'comp.var[i,j]'.")

            comp = attr.value.id
            var = attr.attr

            idx = node.slice
            if isinstance(idx, ast.Tuple):
                indices = tuple(self._eval_ast_node(dim) for dim in idx.elts)
            else:
                # Single index: could be 1D or a full slice
                indices = (self._eval_ast_node(idx),)

            return comp, var, indices
        except Exception as e:
            raise ValueError(f"Invalid expression '{expr}': {e}")

    def _eval_ast_node(self, node):
        if isinstance(node, ast.Slice):
            return slice(
                self._eval_ast_node(node.lower) if node.lower else None,
                self._eval_ast_node(node.upper) if node.upper else None,
                self._eval_ast_node(node.step) if node.step else None,
            )
        elif isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Name) and node.id == "None":
            return None
        else:
            return ast.literal_eval(node)

    def initialize(self):
        # Set up the variables so that they are contiguous
        vars = -np.ones(self.index_pool.counter, dtype=int)

        counter = 0
        for name, comp in self.comp.items():
            for varname, array in comp.vars.items():
                arr = array.ravel()

                for i in range(arr.shape[0]):
                    if vars[arr[i]] == -1:
                        vars[arr[i]] = counter
                        counter += 1

                    arr[i] = vars[arr[i]]

        self.num_variables = counter

        return
    
    def get_vars(self, name, var_name):
        return self.comp[name].get_var(var_name)
    
    def create_opt_problem(self):
        objs = []
        for name, comp in self.comp.items():
            objs.append(comp.create_model(self.module_name))

        return OptimizationProblem(objs)

    def print_indices(self):
        for comp_name, comp in self.comp.items():
            print(f"Component: {comp_name}")
            for varname, array in comp.vars.items():
                print(f"  {varname}:\n{array}")

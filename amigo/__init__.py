from .amigo import Vector, OptimizationProblem, QuasidefCholesky
from .component import Component
from .model import Model
from .optimizer import Optimizer
from .unary_operations import *


def get_include():
    from .amigo import AMIGO_INCLUDE_PATH

    return AMIGO_INCLUDE_PATH


def get_a2d_include():
    from .amigo import A2D_INCLUDE_PATH

    return A2D_INCLUDE_PATH

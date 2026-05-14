import numpy as np
from .expressions import _type_to_str, _str_to_type, _normalize_shape


class Meta:
    @staticmethod
    def get_default_value(name):
        defaults = {
            "shape": (1,),
            "value": 0.0,
            "type": float,
            "lower": -float("inf"),
            "upper": float("inf"),
            "units": None,
            "scale": 1.0,
            "label": None,
        }

        if name in defaults:
            return defaults[name]
        else:
            raise ValueError(f"Unknown meta name {name}")

    def __init__(self, name, var_type, **kwargs):
        self.UNSET = object()
        self.name = name

        options = ["input", "constraint", "output", "data", "objective", "constant"]
        if var_type not in options:
            raise ValueError(f"{var_type} not one of {options}")
        self.var_type = var_type

        # Extract the values and normalize the shapes
        self.shape = _normalize_shape(kwargs.pop("shape", (1,)))

        self.value = self._normalize_shaped_value(
            kwargs.pop("value", self.UNSET), self.shape, "value"
        )
        self.type = kwargs.pop("type", float)
        self.lower = self._normalize_shaped_value(
            kwargs.pop("lower", self.UNSET), self.shape, "lower"
        )
        self.upper = self._normalize_shaped_value(
            kwargs.pop("upper", self.UNSET), self.shape, "upper"
        )
        self.units = self._normalize_shaped_value(
            kwargs.pop("units", self.UNSET), self.shape, "units"
        )
        self.scale = self._normalize_shaped_value(
            kwargs.pop("scale", self.UNSET), self.shape, "scale"
        )
        self.label = kwargs.pop("label", self.UNSET)

        if len(kwargs) > 0:
            raise ValueError(f"Unknown options: {kwargs}")

        # Validate
        self._validate()

    def is_value_set(self, meta_name):
        if self[meta_name] == self.UNSET:
            return False
        return True

    def _expand_scalar(self, x, shape: tuple):
        if shape == (1,) or len(shape) == 0:
            return x

        return tuple(self._expand_scalar(x, shape[1:]) for _ in range(shape[0]))

    def _check_shape(self, x, shape: tuple):
        if len(shape) == 0:
            return True

        if len(x) == shape[0]:
            for a in x:
                return self._check_shape(a, shape[1:])
        else:
            return False

    def _normalize_shaped_value(self, x, shape: tuple, name: str):
        """Get a value that respects the shape"""

        # The value has not been set
        if x == self.UNSET:
            return x

        if shape == None and isinstance(x, (int, float)):
            return x

        if isinstance(x, (int, float)):
            return self._expand_scalar(x, shape)

        if isinstance(x, np.ndarray):
            if x.shape == shape:
                return x
            else:
                ValueError(f"{name} must be a scalar or have shape {shape}")

        if self._check_shape(x, shape):
            return x
        else:
            raise ValueError(f"{name} must be scalar or have shape {shape}")

    def _validate(self):
        if "__" in self.name:
            raise ValueError(f"Name {self.name} cannot contain a double underscore")

        if self.lower is not self.UNSET and self.upper is not self.UNSET:
            if self.shape == None or self.shape == (1,):
                if self.lower > self.upper:
                    raise ValueError("lower bound cannot be greater than upper bound")
            else:
                for lo, hi in zip(self.lower, self.upper):
                    if lo > hi:
                        raise ValueError(
                            "lower bound cannot be greater than upper bound"
                        )

        if (
            self.var_type == "input"
            and self.value is not self.UNSET
            and self.lower is not self.UNSET
            and self.upper is not self.UNSET
        ):
            if self.shape == None or self.shape == (1,):
                if not self.lower <= self.value <= self.upper:
                    raise ValueError("value must be within [lower, upper]")
            else:
                for val, lo, hi in zip(self.value, self.lower, self.upper):
                    if not lo <= val <= hi:
                        raise ValueError("value must be within [lower, upper]")

    def __getitem__(self, name):
        if name == "name":
            return self.name
        elif name == "shape":
            return self.shape
        elif name == "value":
            return self.value
        elif name == "type":
            return self.type
        elif name == "lower":
            return self.lower
        elif name == "upper":
            return self.upper
        elif name == "units":
            return self.units
        elif name == "scale":
            return self.scale
        elif name == "label":
            return self.label

    def __repr__(self):
        return (
            f"Meta(name={self.name!r}, var_type={self.var_type!r}, shape={self.shape},\n"
            f"     value={self.value}, type={self.type.__name__},\n"
            f"     lower={self.lower}, upper={self.upper}, units={self.units!r},\n"
            f"     scale={self.scale}, label={self.label!r})"
        )

    def _serialize_value(self, x):
        if x is self.UNSET:
            return None
        if x is float:
            return "float"
        if x is int:
            return "int"
        return x

    def serialize(self):
        return {
            "name": self.name,
            "var_type": self.var_type,
            "shape": self.shape,
            "value": self._serialize_value(self.value),
            "type": _type_to_str[self.type],
            "lower": self._serialize_value(self.lower),
            "upper": self._serialize_value(self.upper),
            "units": self._serialize_value(self.units),
            "scale": self._serialize_value(self.scale),
            "label": self._serialize_value(self.label),
        }

    @classmethod
    def deserialize(cls, data):
        name = data.pop("name")
        var_type = data.pop("var_type")
        data["type"] = _str_to_type[data["type"]]
        return cls(name, var_type, **data)

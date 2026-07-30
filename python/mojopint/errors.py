class PintError(Exception):
    """Base exception for the covered Pint-compatible API."""


class UndefinedUnitError(AttributeError, PintError):
    def __init__(self, unit_names):
        self.unit_names = unit_names
        name = unit_names if isinstance(unit_names, str) else ", ".join(unit_names)
        super().__init__(f"'{name}' is not defined in the unit registry")


class DimensionalityError(PintError):
    def __init__(self, units1, units2, dim1="", dim2="", extra_msg=""):
        self.units1 = units1
        self.units2 = units2
        self.dim1 = dim1
        self.dim2 = dim2
        message = f"Cannot convert from '{units1}' ({dim1}) to '{units2}' ({dim2})"
        super().__init__(message + extra_msg)


class OffsetUnitCalculusError(PintError):
    def __init__(self, units):
        super().__init__(
            f"Ambiguous operation with offset unit ({units}). "
            "Use a delta unit or convert explicitly."
        )


class DefinitionSyntaxError(PintError):
    pass

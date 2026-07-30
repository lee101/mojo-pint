"""A Pint-compatible unit parser and quantity core accelerated by Mojo."""

from .errors import (
    DefinitionSyntaxError,
    DimensionalityError,
    OffsetUnitCalculusError,
    PintError,
    UndefinedUnitError,
)
from .registry import (
    Quantity,
    Unit,
    UnitRegistry,
    UnitsContainer,
    get_application_registry,
    set_application_registry,
)

__version__ = "0.1.0"

_DEFAULT_REGISTRY = UnitRegistry()
set_application_registry(_DEFAULT_REGISTRY)

__all__ = [
    "DefinitionSyntaxError",
    "DimensionalityError",
    "OffsetUnitCalculusError",
    "PintError",
    "Quantity",
    "Unit",
    "UnitRegistry",
    "UndefinedUnitError",
    "UnitsContainer",
    "get_application_registry",
    "set_application_registry",
]

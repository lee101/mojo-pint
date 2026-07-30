from __future__ import annotations

import math
import numbers
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterator, Mapping

import numpy as np

from ._lib import add_converted_arrays, convert_array, convert_array_inplace
from .errors import (
    DefinitionSyntaxError,
    DimensionalityError,
    OffsetUnitCalculusError,
    UndefinedUnitError,
)

_DIMENSION_NAMES = (
    "[length]",
    "[mass]",
    "[time]",
    "[current]",
    "[temperature]",
    "[substance]",
    "[luminosity]",
)
_ZERO_DIMS = (0.0,) * 7
_TOKEN = re.compile(
    r"\s*(?:(\*\*|[*/^()+-])|"
    r"((?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)|"
    r"([A-Za-z_µμ°Ω%]+))"
)
_DEFAULT_REGISTRY = None


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-13, abs_tol=1e-15)


def _dims_equal(a, b) -> bool:
    return all(_close(x, y) for x, y in zip(a, b))


def _number(value):
    if isinstance(value, (list, tuple)):
        return np.asarray(value)
    return value


def _format_power(value: float) -> str:
    rounded = round(value)
    return str(rounded) if _close(value, rounded) else f"{value:g}"


class UnitsContainer(dict):
    def __str__(self) -> str:
        if not self:
            return "dimensionless"
        positive = []
        negative = []
        for name, exponent in self.items():
            target = positive if exponent > 0 else negative
            power = abs(exponent)
            target.append(name if _close(power, 1) else f"{name} ** {_format_power(power)}")
        top = " * ".join(positive) if positive else "1"
        return top if not negative else f"{top} / {' / '.join(negative)}"

    __repr__ = __str__


@dataclass(frozen=True)
class _UnitDef:
    name: str
    symbol: str
    dimensions: tuple[float, ...]
    scale: float
    offset: float = 0.0
    prefixable: bool = False
    delta_name: str | None = None


class Unit:
    __array_priority__ = 1000

    def __init__(self, units, _registry=None):
        registry = _registry or _DEFAULT_REGISTRY
        if registry is None:
            raise RuntimeError("no application unit registry is configured")
        parsed = registry.parse_units(units)
        self._registry = registry
        self._dimensions = parsed._dimensions
        self._scale = parsed._scale
        self._offset = parsed._offset
        self._terms = parsed._terms.copy()
        self._delta_name = parsed._delta_name

    @classmethod
    def _create(
        cls,
        registry,
        dimensions=_ZERO_DIMS,
        scale=1.0,
        offset=0.0,
        terms=None,
        delta_name=None,
    ):
        value = object.__new__(cls)
        value._registry = registry
        value._dimensions = tuple(float(x) for x in dimensions)
        value._scale = float(scale)
        value._offset = float(offset)
        value._terms = dict(terms or {})
        value._delta_name = delta_name
        return value

    @property
    def dimensionality(self) -> UnitsContainer:
        return UnitsContainer(
            (name, exponent)
            for name, exponent in zip(_DIMENSION_NAMES, self._dimensions)
            if not _close(exponent, 0)
        )

    @property
    def dimensionless(self) -> bool:
        return _dims_equal(self._dimensions, _ZERO_DIMS)

    @property
    def _is_delta(self) -> bool:
        return len(self._terms) == 1 and next(iter(self._terms)).startswith("delta_")

    def is_compatible_with(self, other, *contexts, **ctx_kwargs) -> bool:
        try:
            candidate = (
                other.units
                if isinstance(other, Quantity)
                else self._registry.parse_units(other)
            )
        except (UndefinedUnitError, TypeError, ValueError):
            return False
        return _dims_equal(self._dimensions, candidate._dimensions)

    def _multiplicative(self):
        if self._offset != 0:
            raise OffsetUnitCalculusError(str(self))

    def _combine(self, other, sign: float):
        if isinstance(other, str):
            other = self._registry.parse_units(other)
        if isinstance(other, Unit):
            self._multiplicative()
            other._multiplicative()
            terms = self._terms.copy()
            for name, power in other._terms.items():
                terms[name] = terms.get(name, 0.0) + sign * power
                if _close(terms[name], 0):
                    del terms[name]
            dimensions = tuple(
                left + sign * right
                for left, right in zip(self._dimensions, other._dimensions)
            )
            scale = self._scale * (
                other._scale if sign > 0 else 1.0 / other._scale
            )
            return Unit._create(self._registry, dimensions, scale, terms=terms)
        if sign > 0:
            return Quantity(other, self, _registry=self._registry)
        return Quantity(1.0 / other, self, _registry=self._registry)

    def __mul__(self, other):
        return self._combine(other, 1.0)

    def __rmul__(self, other):
        return Quantity(other, self, _registry=self._registry)

    def __truediv__(self, other):
        return self._combine(other, -1.0)

    def __rtruediv__(self, other):
        return Quantity(other, self ** -1, _registry=self._registry)

    def __pow__(self, power):
        if not isinstance(power, numbers.Real):
            return NotImplemented
        self._multiplicative()
        terms = {
            name: exponent * float(power)
            for name, exponent in self._terms.items()
            if not _close(exponent * float(power), 0)
        }
        return Unit._create(
            self._registry,
            tuple(value * float(power) for value in self._dimensions),
            self._scale ** float(power),
            terms=terms,
        )

    def __eq__(self, other):
        if not isinstance(other, Unit):
            try:
                other = self._registry.parse_units(other)
            except Exception:
                return False
        return (
            _dims_equal(self._dimensions, other._dimensions)
            and _close(self._scale, other._scale)
            and _close(self._offset, other._offset)
        )

    def __hash__(self):
        values = tuple(round(x, 14) for x in self._dimensions)
        return hash((values, round(self._scale, 14), round(self._offset, 14)))

    def __str__(self):
        if not self._terms:
            return "dimensionless"
        positive = []
        negative = []
        for name in sorted(self._terms):
            exponent = self._terms[name]
            target = positive if exponent > 0 else negative
            power = abs(exponent)
            target.append(name if _close(power, 1) else f"{name} ** {_format_power(power)}")
        numerator = " * ".join(positive) if positive else "1"
        return numerator if not negative else f"{numerator} / {' / '.join(negative)}"

    def __repr__(self):
        return f"<Unit('{self}')>"

    def __format__(self, spec):
        if "~" not in spec:
            return str(self)
        pieces = []
        for name, exponent in sorted(self._terms.items()):
            symbol = self._registry._definitions[name].symbol
            pieces.append(symbol if _close(exponent, 1) else f"{symbol} ** {_format_power(exponent)}")
        return " * ".join(pieces) if pieces else ""


class Quantity:
    __array_priority__ = 1000

    def __init__(self, value, units=None, _registry=None):
        registry = _registry or _DEFAULT_REGISTRY
        if registry is None:
            raise RuntimeError("no application unit registry is configured")
        if isinstance(value, Quantity):
            if units is None:
                self._magnitude = value.magnitude
                self._units = value.units
            else:
                converted = value.to(units)
                self._magnitude = converted.magnitude
                self._units = converted.units
            self._registry = registry
            return
        self._registry = registry
        self._units = registry.parse_units(units or "")
        self._magnitude = _number(value)

    @property
    def magnitude(self):
        return self._magnitude

    @property
    def m(self):
        return self._magnitude

    @property
    def units(self):
        return self._units

    @property
    def u(self):
        return self._units

    @property
    def dimensionality(self):
        return self._units.dimensionality

    @property
    def dimensionless(self):
        return self._units.dimensionless

    @property
    def shape(self):
        return np.shape(self._magnitude)

    @property
    def ndim(self):
        return np.ndim(self._magnitude)

    def __len__(self):
        return len(self._magnitude)

    def __iter__(self) -> Iterator["Quantity"]:
        for item in self._magnitude:
            yield Quantity(item, self._units, _registry=self._registry)

    def __getitem__(self, item):
        return Quantity(self._magnitude[item], self._units, _registry=self._registry)

    def _conversion(self, other):
        target = self._registry.parse_units(other)
        if not _dims_equal(self._units._dimensions, target._dimensions):
            raise DimensionalityError(
                self._units,
                target,
                self._units.dimensionality,
                target.dimensionality,
            )
        if (self._units._offset and target._is_delta) or (
            target._offset and self._units._is_delta
        ):
            raise DimensionalityError(
                self._units,
                target,
                self._units.dimensionality,
                target.dimensionality,
            )
        scale = self._units._scale / target._scale
        shift = (self._units._offset - target._offset) / target._scale
        return target, scale, shift

    def to(self, other=None, *contexts, **ctx_kwargs):
        target, scale, shift = self._conversion(other or "")
        if isinstance(self._magnitude, np.ndarray):
            magnitude = convert_array(self._magnitude, scale, shift)
        else:
            magnitude = self._magnitude * scale + shift
        return Quantity(magnitude, target, _registry=self._registry)

    def ito(self, other=None, *contexts, **ctx_kwargs) -> None:
        target, scale, shift = self._conversion(other or "")
        if isinstance(self._magnitude, np.ndarray):
            self._magnitude = convert_array_inplace(
                self._magnitude, scale, shift
            )
        else:
            self._magnitude = self._magnitude * scale + shift
        self._units = target

    def m_as(self, units):
        return self.to(units).magnitude

    def to_base_units(self):
        target = Unit._create(
            self._registry,
            self._units._dimensions,
            terms=self._registry._base_terms(self._units._dimensions),
        )
        return self.to(target)

    def to_reduced_units(self):
        return self.to_base_units()

    def is_compatible_with(self, other, *contexts, **ctx_kwargs):
        return self._units.is_compatible_with(other, *contexts, **ctx_kwargs)

    def check(self, dimension):
        return self._units.is_compatible_with(
            self._registry._parse_dimension_expression(dimension)
        )

    def _coerce_add(self, other):
        if isinstance(other, Quantity):
            return other
        if self.dimensionless:
            return Quantity(other, "", _registry=self._registry)
        if other == 0:
            return Quantity(0, self._units, _registry=self._registry)
        raise DimensionalityError(self._units, "dimensionless")

    def __add__(self, other):
        other = self._coerce_add(other)
        if self._units._offset:
            if not other.units._is_delta:
                raise OffsetUnitCalculusError(f"{self._units}, {other.units}")
            delta = other.magnitude * other.units._scale / self._units._scale
            return Quantity(
                self._magnitude + delta, self._units, _registry=self._registry
            )
        if other.units._offset:
            if self._units._is_delta:
                return other + self
            raise OffsetUnitCalculusError(f"{self._units}, {other.units}")
        target, scale, shift = other._conversion(self._units)
        if isinstance(self._magnitude, np.ndarray) and isinstance(
            other.magnitude, np.ndarray
        ):
            magnitude = add_converted_arrays(
                self._magnitude, other.magnitude, scale, shift
            )
            return Quantity(magnitude, target, _registry=self._registry)
        converted = other.to(self._units)
        return Quantity(
            self._magnitude + converted.magnitude,
            self._units,
            _registry=self._registry,
        )

    __radd__ = __add__

    def __sub__(self, other):
        other = self._coerce_add(other)
        if self._units._offset and other.units._is_delta:
            delta = other.magnitude * other.units._scale / self._units._scale
            return Quantity(
                self._magnitude - delta, self._units, _registry=self._registry
            )
        if self._units._offset:
            si_difference = (
                self._magnitude * self._units._scale
                + self._units._offset
                - other._magnitude * other.units._scale
                - other.units._offset
            )
            delta_name = self._units._delta_name or "kelvin"
            target = self._registry.parse_units(delta_name)
            return Quantity(
                si_difference / target._scale,
                target,
                _registry=self._registry,
            )
        if other.units._offset:
            converted = other.to(self._units)
            return Quantity(
                self._magnitude - converted.magnitude,
                self._units,
                _registry=self._registry,
            )
        target, scale, shift = other._conversion(self._units)
        if isinstance(self._magnitude, np.ndarray) and isinstance(
            other.magnitude, np.ndarray
        ):
            magnitude = add_converted_arrays(
                self._magnitude, other.magnitude, scale, shift, -1.0
            )
            return Quantity(magnitude, target, _registry=self._registry)
        converted = other.to(self._units)
        return Quantity(
            self._magnitude - converted.magnitude,
            self._units,
            _registry=self._registry,
        )

    def __rsub__(self, other):
        return Quantity(other, "", _registry=self._registry) - self

    def __mul__(self, other):
        if isinstance(other, Quantity):
            units = self._units * other.units
            magnitude = self._magnitude * other.magnitude
        elif isinstance(other, Unit):
            units = self._units * other
            magnitude = self._magnitude
        else:
            if self._units._offset:
                raise OffsetUnitCalculusError(str(self._units))
            units = self._units
            magnitude = self._magnitude * other
        return Quantity(magnitude, units, _registry=self._registry)

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, Quantity):
            units = self._units / other.units
            magnitude = self._magnitude / other.magnitude
        elif isinstance(other, Unit):
            units = self._units / other
            magnitude = self._magnitude
        else:
            if self._units._offset:
                raise OffsetUnitCalculusError(str(self._units))
            units = self._units
            magnitude = self._magnitude / other
        return Quantity(magnitude, units, _registry=self._registry)

    def __rtruediv__(self, other):
        return Quantity(
            other / self._magnitude,
            self._units ** -1,
            _registry=self._registry,
        )

    def __pow__(self, power):
        if isinstance(power, Quantity):
            if not power.dimensionless:
                raise DimensionalityError(power.units, "dimensionless")
            power = power.magnitude
        return Quantity(
            self._magnitude ** power,
            self._units ** power,
            _registry=self._registry,
        )

    def __neg__(self):
        return Quantity(-self._magnitude, self._units, _registry=self._registry)

    def __pos__(self):
        return self

    def __abs__(self):
        return Quantity(abs(self._magnitude), self._units, _registry=self._registry)

    def _compare(self, other, operation):
        other = self._coerce_add(other).to(self._units)
        return operation(self._magnitude, other.magnitude)

    def __lt__(self, other):
        return self._compare(other, lambda a, b: a < b)

    def __le__(self, other):
        return self._compare(other, lambda a, b: a <= b)

    def __gt__(self, other):
        return self._compare(other, lambda a, b: a > b)

    def __ge__(self, other):
        return self._compare(other, lambda a, b: a >= b)

    def __eq__(self, other):
        try:
            converted = self._coerce_add(other).to(self._units)
            result = np.equal(self._magnitude, converted.magnitude)
            return bool(result) if np.ndim(result) == 0 else result
        except (DimensionalityError, TypeError, ValueError):
            return False

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self._magnitude, dtype=dtype)

    def __str__(self):
        return f"{self._magnitude} {self._units}"

    def __repr__(self):
        return f"<Quantity({self._magnitude!r}, '{self._units}')>"

    def __format__(self, spec):
        if "~" in spec:
            clean = spec.replace("~", "")
            return f"{format(self._magnitude, clean)} {format(self._units, '~')}"
        return str(self)


class _ExpressionParser:
    def __init__(self, registry, text: str):
        self.registry = registry
        self.tokens = self._tokenize(text)
        self.index = 0

    @staticmethod
    def _tokenize(text):
        text = (
            text.strip()
            .replace("·", "*")
            .replace("×", "*")
            .replace(" per ", " / ")
        )
        text = re.sub(r"\bsquare\s+([A-Za-z_µμ°Ω%]+)", r"\1 ** 2", text)
        text = re.sub(r"\bcubic\s+([A-Za-z_µμ°Ω%]+)", r"\1 ** 3", text)
        tokens = []
        position = 0
        while position < len(text):
            match = _TOKEN.match(text, position)
            if not match:
                raise ValueError(f"invalid unit expression near {text[position:]!r}")
            operator, number, name = match.groups()
            tokens.append(operator or number or name)
            position = match.end()
        return tokens

    def peek(self):
        return self.tokens[self.index] if self.index < len(self.tokens) else None

    def take(self):
        token = self.peek()
        self.index += 1
        return token

    def parse(self):
        if not self.tokens:
            return 1.0, self.registry._dimensionless()
        coefficient, unit = self.product()
        if self.peek() is not None:
            raise ValueError(f"unexpected token {self.peek()!r}")
        return coefficient, unit

    def product(self):
        coefficient, unit = self.power()
        while self.peek() is not None and self.peek() != ")":
            operator = "*"
            if self.peek() in ("*", "/"):
                operator = self.take()
            right_coefficient, right_unit = self.power()
            if operator == "*":
                coefficient *= right_coefficient
                unit = unit * right_unit
            else:
                coefficient /= right_coefficient
                unit = unit / right_unit
        return coefficient, unit

    def power(self):
        coefficient, unit = self.atom()
        if self.peek() in ("^", "**"):
            self.take()
            sign = 1.0
            if self.peek() in ("+", "-"):
                sign = -1.0 if self.take() == "-" else 1.0
            token = self.take()
            if token is None or not re.fullmatch(
                r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", token
            ):
                raise ValueError("unit exponent must be numeric")
            exponent = sign * float(token)
            coefficient = coefficient**exponent
            unit = unit**exponent
        return coefficient, unit

    def atom(self):
        token = self.take()
        if token is None:
            raise ValueError("incomplete unit expression")
        if token == "(":
            value = self.product()
            if self.take() != ")":
                raise ValueError("unclosed parenthesis in unit expression")
            return value
        if token in ("+", "-"):
            coefficient, unit = self.atom()
            return (-coefficient if token == "-" else coefficient), unit
        if re.fullmatch(r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", token):
            return float(token), self.registry._dimensionless()
        if token in ("*", "/", ")", "^", "**"):
            raise ValueError(f"unexpected token {token!r}")
        return 1.0, self.registry._resolve(token)


class UnitRegistry:
    def __init__(self, *args, **kwargs):
        self.autoconvert_offset_to_baseunit = kwargs.pop(
            "autoconvert_offset_to_baseunit", False
        )
        self._definitions: dict[str, _UnitDef] = {}
        self._aliases: dict[str, str] = {}
        self._install_defaults()

    def _add(
        self,
        name,
        symbol,
        aliases,
        dimensions,
        scale=1.0,
        offset=0.0,
        prefixable=False,
        delta_name=None,
    ):
        definition = _UnitDef(
            name,
            symbol,
            tuple(float(x) for x in dimensions),
            float(scale),
            float(offset),
            prefixable,
            delta_name,
        )
        self._definitions[name] = definition
        for alias in (name, symbol, *aliases):
            self._aliases[alias] = name

    def _install_defaults(self):
        L, M, T, I, K, N, J = (
            (1, 0, 0, 0, 0, 0, 0),
            (0, 1, 0, 0, 0, 0, 0),
            (0, 0, 1, 0, 0, 0, 0),
            (0, 0, 0, 1, 0, 0, 0),
            (0, 0, 0, 0, 1, 0, 0),
            (0, 0, 0, 0, 0, 1, 0),
            (0, 0, 0, 0, 0, 0, 1),
        )
        Z = _ZERO_DIMS
        add = self._add
        add("meter", "m", ("metre",), L, prefixable=True)
        add("second", "s", ("sec",), T, prefixable=True)
        add("kilogram", "kg", (), M)
        add("gram", "g", (), M, 1e-3, prefixable=True)
        add("ampere", "A", ("amp",), I, prefixable=True)
        add("kelvin", "K", (), K, prefixable=True)
        add("mole", "mol", (), N, prefixable=True)
        add("candela", "cd", (), J, prefixable=True)
        add("radian", "rad", (), Z)
        add("steradian", "sr", (), Z)
        add("degree", "deg", ("degree_angle", "°"), Z, math.pi / 180)
        add("revolution", "turn", ("cycle",), Z, 2 * math.pi)
        add("percent", "%", ("pct",), Z, 0.01)
        add("parts_per_million", "ppm", (), Z, 1e-6)

        derived = {
            "hertz": ("Hz", (0, 0, -1, 0, 0, 0, 0), 1.0, ("cycle_per_second",)),
            "newton": ("N", (1, 1, -2, 0, 0, 0, 0), 1.0, ()),
            "pascal": ("Pa", (-1, 1, -2, 0, 0, 0, 0), 1.0, ()),
            "joule": ("J", (2, 1, -2, 0, 0, 0, 0), 1.0, ()),
            "watt": ("W", (2, 1, -3, 0, 0, 0, 0), 1.0, ()),
            "coulomb": ("C", (0, 0, 1, 1, 0, 0, 0), 1.0, ()),
            "volt": ("V", (2, 1, -3, -1, 0, 0, 0), 1.0, ()),
            "farad": ("F", (-2, -1, 4, 2, 0, 0, 0), 1.0, ()),
            "ohm": ("Ω", (2, 1, -3, -2, 0, 0, 0), 1.0, ("ohm",)),
            "siemens": ("S", (-2, -1, 3, 2, 0, 0, 0), 1.0, ()),
            "weber": ("Wb", (2, 1, -2, -1, 0, 0, 0), 1.0, ()),
            "tesla": ("T", (0, 1, -2, -1, 0, 0, 0), 1.0, ()),
            "henry": ("H", (2, 1, -2, -2, 0, 0, 0), 1.0, ()),
            "lumen": ("lm", (0, 0, 0, 0, 0, 0, 1), 1.0, ()),
            "lux": ("lx", (-2, 0, 0, 0, 0, 0, 1), 1.0, ()),
            "becquerel": ("Bq", (0, 0, -1, 0, 0, 0, 0), 1.0, ()),
            "gray": ("Gy", (2, 0, -2, 0, 0, 0, 0), 1.0, ()),
            "sievert": ("Sv", (2, 0, -2, 0, 0, 0, 0), 1.0, ()),
            "katal": ("kat", (0, 0, -1, 0, 0, 1, 0), 1.0, ()),
        }
        for name, (symbol, dims, scale, aliases) in derived.items():
            add(name, symbol, aliases, dims, scale, prefixable=True)

        for name, symbol, aliases, scale in (
            ("minute", "min", (), 60),
            ("hour", "h", ("hr",), 3600),
            ("day", "d", (), 86400),
            ("week", "wk", (), 604800),
            ("year", "yr", ("julian_year",), 31557600),
        ):
            add(name, symbol, aliases, T, scale)
        for name, symbol, aliases, scale in (
            ("inch", "in", ("inches",), 0.0254),
            ("foot", "ft", ("feet",), 0.3048),
            ("yard", "yd", (), 0.9144),
            ("mile", "mi", (), 1609.344),
            ("nautical_mile", "nmi", (), 1852),
            ("angstrom", "Å", (), 1e-10),
            ("astronomical_unit", "au", (), 149597870700),
            ("light_year", "ly", (), 9460730472580800),
            ("parsec", "pc", (), 3.0856775814671916e16),
        ):
            add(name, symbol, aliases, L, scale)
        for name, symbol, aliases, scale in (
            ("ounce", "oz", (), 0.028349523125),
            ("pound", "lb", ("lbm",), 0.45359237),
            ("stone", "st", (), 6.35029318),
            ("short_ton", "ton", (), 907.18474),
            ("metric_ton", "t", ("tonne",), 1000),
        ):
            add(name, symbol, aliases, M, scale)

        volume = (3, 0, 0, 0, 0, 0, 0)
        add("liter", "L", ("litre", "l"), volume, 1e-3, prefixable=True)
        add("gallon", "gal", ("US_gallon",), volume, 0.003785411784)
        add("quart", "qt", (), volume, 0.000946352946)
        add("pint", "pt", (), volume, 0.000473176473)
        add("fluid_ounce", "floz", (), volume, 2.95735295625e-5)

        pressure = (-1, 1, -2, 0, 0, 0, 0)
        add("bar", "bar", (), pressure, 1e5, prefixable=True)
        add("standard_atmosphere", "atm", ("atmosphere",), pressure, 101325)
        add("torr", "Torr", ("mmHg",), pressure, 101325 / 760)
        add("psi", "psi", ("pound_force_per_square_inch",), pressure, 6894.757293168)

        energy = (2, 1, -2, 0, 0, 0, 0)
        power = (2, 1, -3, 0, 0, 0, 0)
        add("calorie", "cal", (), energy, 4.184, prefixable=True)
        add("electron_volt", "eV", (), energy, 1.602176634e-19, prefixable=True)
        add("british_thermal_unit", "Btu", (), energy, 1055.056)
        add("watt_hour", "Wh", (), energy, 3600)
        add("kilowatt_hour", "kWh", (), energy, 3.6e6)
        add("horsepower", "hp", (), power, 745.6998715822702)
        add("knot", "kn", (), (1, 0, -1, 0, 0, 0, 0), 1852 / 3600)
        add("revolutions_per_minute", "rpm", (), (0, 0, -1, 0, 0, 0, 0), 2 * math.pi / 60)

        add(
            "degree_Celsius",
            "°C",
            ("degC", "celsius"),
            K,
            1,
            273.15,
            delta_name="delta_degree_Celsius",
        )
        add(
            "delta_degree_Celsius",
            "Δ°C",
            ("delta_degC",),
            K,
            1,
        )
        add(
            "degree_Fahrenheit",
            "°F",
            ("degF", "fahrenheit"),
            K,
            5 / 9,
            255.3722222222222,
            delta_name="delta_degree_Fahrenheit",
        )
        add(
            "delta_degree_Fahrenheit",
            "Δ°F",
            ("delta_degF",),
            K,
            5 / 9,
        )
        add("degree_Rankine", "°R", ("degR", "rankine"), K, 5 / 9)

    def _dimensionless(self):
        return Unit._create(self)

    def _unit_from_definition(self, definition):
        return Unit._create(
            self,
            definition.dimensions,
            definition.scale,
            definition.offset,
            {definition.name: 1.0},
            definition.delta_name,
        )

    @lru_cache(maxsize=1024)
    def _resolve(self, name):
        if name == "dimensionless":
            return self._dimensionless()
        canonical = self._aliases.get(name)
        if canonical:
            return self._unit_from_definition(self._definitions[canonical])
        if len(name) > 2 and name.endswith("s"):
            singular = name[:-1]
            canonical = self._aliases.get(singular)
            if canonical:
                return self._unit_from_definition(self._definitions[canonical])

        prefixes = (
            ("quetta", "Q", 1e30),
            ("ronna", "R", 1e27),
            ("yotta", "Y", 1e24),
            ("zetta", "Z", 1e21),
            ("exa", "E", 1e18),
            ("peta", "P", 1e15),
            ("tera", "T", 1e12),
            ("giga", "G", 1e9),
            ("mega", "M", 1e6),
            ("kilo", "k", 1e3),
            ("hecto", "h", 1e2),
            ("deca", "da", 1e1),
            ("deci", "d", 1e-1),
            ("centi", "c", 1e-2),
            ("milli", "m", 1e-3),
            ("micro", "u", 1e-6),
            ("micro", "µ", 1e-6),
            ("micro", "μ", 1e-6),
            ("nano", "n", 1e-9),
            ("pico", "p", 1e-12),
            ("femto", "f", 1e-15),
            ("atto", "a", 1e-18),
            ("zepto", "z", 1e-21),
            ("yocto", "y", 1e-24),
            ("ronto", "r", 1e-27),
            ("quecto", "q", 1e-30),
        )
        for prefix_name, prefix_symbol, factor in prefixes:
            for prefix in (prefix_name, prefix_symbol):
                if not name.startswith(prefix) or len(name) == len(prefix):
                    continue
                remainder = name[len(prefix) :]
                base_name = self._aliases.get(remainder)
                if not base_name and remainder.endswith("s"):
                    base_name = self._aliases.get(remainder[:-1])
                if not base_name:
                    continue
                base = self._definitions[base_name]
                if not base.prefixable:
                    continue
                canonical_name = prefix_name + base.name
                definition = self._definitions.get(canonical_name)
                if definition is None:
                    definition = _UnitDef(
                        canonical_name,
                        prefix_symbol + base.symbol,
                        base.dimensions,
                        factor * base.scale,
                        0.0,
                    )
                    self._definitions[canonical_name] = definition
                    self._aliases[name] = canonical_name
                return self._unit_from_definition(definition)
        raise UndefinedUnitError(name)

    @lru_cache(maxsize=4096)
    def _parse_cached(self, expression):
        return _ExpressionParser(self, expression).parse()

    def parse_expression(self, input_string, *args, **kwargs):
        if isinstance(input_string, Quantity):
            return input_string
        if not isinstance(input_string, str):
            raise TypeError("input_string must be a string")
        coefficient, unit = self._parse_cached(input_string)
        if _close(coefficient, round(coefficient)):
            coefficient = int(round(coefficient))
        return Quantity(coefficient, unit, _registry=self)

    __call__ = parse_expression

    def parse_units(self, input_string, as_delta=None, case_sensitive=None):
        if input_string is None or input_string == "":
            return self._dimensionless()
        if isinstance(input_string, Unit):
            return input_string
        if isinstance(input_string, Quantity):
            if np.ndim(input_string.magnitude) or input_string.magnitude != 1:
                raise ValueError("quantity used as a unit must have magnitude 1")
            return input_string.units
        if not isinstance(input_string, str):
            raise TypeError("units must be a string or Unit")
        coefficient, unit = self._parse_cached(input_string)
        if not _close(coefficient, 1):
            unit = Unit._create(
                self,
                unit._dimensions,
                unit._scale * coefficient,
                unit._offset,
                unit._terms,
                unit._delta_name,
            )
        return unit

    def Quantity(self, value, units=None):
        if isinstance(value, str) and units is None:
            return self.parse_expression(value)
        return Quantity(value, units, _registry=self)

    def Unit(self, units):
        return self.parse_units(units)

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self._resolve(name)
        except UndefinedUnitError as error:
            raise error from None

    def __getitem__(self, name):
        return self._resolve(name)

    def __contains__(self, name):
        try:
            self._resolve(name)
            return True
        except UndefinedUnitError:
            return False

    def get_name(self, name):
        unit = self._resolve(name)
        return next(iter(unit._terms))

    def get_symbol(self, name):
        return self._definitions[self.get_name(name)].symbol

    def get_compatible_units(self, input_units, *contexts):
        unit = self.parse_units(input_units)
        return {
            self._unit_from_definition(definition)
            for definition in self._definitions.values()
            if _dims_equal(unit._dimensions, definition.dimensions)
        }

    def _base_terms(self, dimensions):
        names = (
            "meter",
            "kilogram",
            "second",
            "ampere",
            "kelvin",
            "mole",
            "candela",
        )
        return {
            name: exponent
            for name, exponent in zip(names, dimensions)
            if not _close(exponent, 0)
        }

    def get_base_units(self, input_units, check_nonmult=True, system=None):
        unit = self.parse_units(input_units)
        base = Unit._create(
            self, unit._dimensions, terms=self._base_terms(unit._dimensions)
        )
        return unit._scale, base

    def _parse_dimension_expression(self, expression):
        if isinstance(expression, Unit):
            return expression
        text = str(expression)
        replacements = {
            "[length]": "meter",
            "[mass]": "kilogram",
            "[time]": "second",
            "[current]": "ampere",
            "[temperature]": "kelvin",
            "[substance]": "mole",
            "[luminosity]": "candela",
        }
        for dimension, unit in replacements.items():
            text = text.replace(dimension, unit)
        return self.parse_units(text)

    def define(self, definition):
        if not isinstance(definition, str):
            raise DefinitionSyntaxError("only string definitions are supported")
        fields = [field.strip() for field in definition.split("=")]
        if len(fields) < 2 or not fields[0]:
            raise DefinitionSyntaxError(definition)
        name = fields[0]
        aliases = tuple(field for field in fields[2:] if field)
        coefficient, reference = self._parse_cached(fields[1])
        if reference._offset:
            raise DefinitionSyntaxError("definitions cannot derive from offset units")
        symbol = aliases[0] if aliases else name
        self._add(
            name,
            symbol,
            aliases[1:],
            reference._dimensions,
            coefficient * reference._scale,
        )
        self._parse_cached.cache_clear()
        self._resolve.cache_clear()


def set_application_registry(registry):
    global _DEFAULT_REGISTRY
    _DEFAULT_REGISTRY = registry


def get_application_registry():
    return _DEFAULT_REGISTRY

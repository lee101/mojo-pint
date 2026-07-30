import numpy as np
import pytest

import mojopint as mp

pint = pytest.importorskip("pint")


@pytest.fixture(scope="module")
def registries():
    return mp.UnitRegistry(), pint.UnitRegistry()


def dims(quantity):
    return {name: float(power) for name, power in quantity.dimensionality.items()}


@pytest.mark.parametrize(
    "expression",
    [
        "m/s^2",
        "kg m / s^2",
        "9.81 meter / second ** 2",
        "square meter / second",
        "(newton * meter) / second",
        "meter ** 0.5",
        "rpm",
        "percent",
        "kW h",
        "kWh",
        "milligram / milliliter",
        "1e3 * micrometer",
    ],
)
def test_expression_parsing_matches_pint(registries, expression):
    ours, upstream = registries
    a, b = ours(expression), upstream(expression)
    assert a.magnitude == pytest.approx(b.magnitude)
    assert dims(a) == dims(b)
    assert a.to_base_units().magnitude == pytest.approx(
        b.to_base_units().magnitude
    )


@pytest.mark.parametrize(
    "source,destination,value",
    [
        ("mile", "kilometer", 3.25),
        ("foot", "centimeter", 11.0),
        ("gallon", "liter", 2.0),
        ("pound", "kilogram", 160.0),
        ("knot", "meter / second", 17.0),
        ("psi", "kilopascal", 35.0),
        ("atmosphere", "pascal", 0.85),
        ("calorie", "joule", 125.0),
        ("electron_volt", "joule", 6.2e18),
        ("degree", "radian", 180.0),
        ("rpm", "1 / second", 3600.0),
        ("megawatt", "horsepower", 1.0),
    ],
)
def test_scalar_conversions_match_pint(registries, source, destination, value):
    ours, upstream = registries
    assert ours.Quantity(value, source).m_as(destination) == pytest.approx(
        upstream.Quantity(value, source).m_as(destination), rel=2e-12
    )


@pytest.mark.parametrize(
    "source,destination",
    [
        ("meter", "inch"),
        ("kilometer / hour", "meter / second"),
        ("degree_Celsius", "degree_Fahrenheit"),
        ("degree_Fahrenheit", "kelvin"),
    ],
)
def test_bulk_array_conversion_matches_pint(registries, source, destination):
    ours, upstream = registries
    values = np.linspace(-40, 500, 100_003)
    got = ours.Quantity(values, source).m_as(destination)
    expected = upstream.Quantity(values, source).m_as(destination)
    assert got.shape == values.shape
    assert np.allclose(got, expected, rtol=2e-13, atol=2e-12)


def test_temperature_points_and_differences_match_pint(registries):
    ours, upstream = registries
    assert ours.Quantity(0, "degC").m_as("degF") == pytest.approx(
        upstream.Quantity(0, "degC").m_as("degF")
    )
    assert ours.Quantity(100, "degC").m_as("K") == pytest.approx(
        upstream.Quantity(100, "degC").m_as("K")
    )
    a = ours.Quantity(20, "degC") - ours.Quantity(10, "degC")
    b = upstream.Quantity(20, "degC") - upstream.Quantity(10, "degC")
    assert a.magnitude == pytest.approx(b.magnitude)
    assert str(a.units) == str(b.units)
    a = ours.Quantity(20, "degC") + ours.Quantity(10, "delta_degC")
    b = upstream.Quantity(20, "degC") + upstream.Quantity(10, "delta_degC")
    assert a.magnitude == pytest.approx(b.magnitude)
    assert str(a.units) == str(b.units)


def test_offset_errors_match_pint(registries):
    ours, upstream = registries
    with pytest.raises(mp.OffsetUnitCalculusError):
        ours.Quantity(10, "degC") + ours.Quantity(5, "degC")
    with pytest.raises(pint.OffsetUnitCalculusError):
        upstream.Quantity(10, "degC") + upstream.Quantity(5, "degC")
    with pytest.raises(mp.DimensionalityError):
        ours.Quantity(10, "delta_degC").to("degC")
    with pytest.raises(pint.DimensionalityError):
        upstream.Quantity(10, "delta_degC").to("degC")


def test_quantity_arithmetic_and_reduction_match_pint(registries):
    ours, upstream = registries
    a = 12 * ours.newton * (3 * ours.meter)
    b = 12 * upstream.newton * (3 * upstream.meter)
    assert a.to("joule").magnitude == pytest.approx(b.to("joule").magnitude)
    speed_a = 100 * ours.kilometer / (2 * ours.hour)
    speed_b = 100 * upstream.kilometer / (2 * upstream.hour)
    assert speed_a.to("m/s").magnitude == pytest.approx(
        speed_b.to("m/s").magnitude
    )
    assert dims(speed_a) == dims(speed_b)


def test_addition_converts_rhs_to_lhs(registries):
    ours, upstream = registries
    a = ours.Quantity(1, "meter") + ours.Quantity(25, "centimeter")
    b = upstream.Quantity(1, "meter") + upstream.Quantity(25, "centimeter")
    assert a.magnitude == pytest.approx(b.magnitude)
    assert str(a.units) == str(b.units)


def test_incompatible_conversion_raises(registries):
    ours, upstream = registries
    with pytest.raises(mp.DimensionalityError):
        ours.Quantity(1, "meter").to("second")
    with pytest.raises(pint.DimensionalityError):
        upstream.Quantity(1, "meter").to("second")


def test_undefined_unit_raises(registries):
    ours, upstream = registries
    with pytest.raises(mp.UndefinedUnitError):
        ours("definitely_not_a_unit")
    with pytest.raises(pint.UndefinedUnitError):
        upstream("definitely_not_a_unit")


def test_unit_compatibility_and_check_match_pint(registries):
    ours, upstream = registries
    pairs = [
        ("newton", "kg*m/s^2"),
        ("joule", "newton*meter"),
        ("hertz", "1/second"),
        ("liter", "meter^3"),
        ("meter", "second"),
    ]
    for left, right in pairs:
        assert ours.Unit(left).is_compatible_with(right) == upstream.Unit(
            left
        ).is_compatible_with(right)
    assert ours.Quantity(2, "m/s").check("[length] / [time]")
    assert not ours.Quantity(2, "m/s").check("[mass]")


def test_registry_attribute_plural_and_symbol_resolution(registries):
    ours, upstream = registries
    for name in ("meters", "km", "MPa", "µm", "feet", "inches", "Btu", "ms"):
        assert dims(ours(name)) == dims(upstream(name))
        assert ours(name).to_base_units().magnitude == pytest.approx(
            upstream(name).to_base_units().magnitude
        )


def test_every_bundled_unit_matches_pint(registries):
    ours, upstream = registries
    for name, definition in ours._definitions.items():
        ours_unit = ours.Unit(name)
        upstream_name = name if name in upstream else definition.symbol
        upstream_unit = upstream.Unit(upstream_name)
        assert dims(ours.Quantity(1, ours_unit)) == dims(
            upstream.Quantity(1, upstream_unit)
        ), name
        if definition.offset:
            ours_value = ours.Quantity(1, ours_unit).m_as("kelvin")
            upstream_value = upstream.Quantity(1, upstream_unit).m_as("kelvin")
        else:
            ours_value = ours.Quantity(1, ours_unit).to_base_units().magnitude
            upstream_value = (
                upstream.Quantity(1, upstream_unit).to_base_units().magnitude
            )
        assert ours_value == pytest.approx(upstream_value, rel=2e-12), name


def test_every_decimal_prefix_matches_pint(registries):
    ours, upstream = registries
    prefixes = (
        "quetta",
        "ronna",
        "yotta",
        "zetta",
        "exa",
        "peta",
        "tera",
        "giga",
        "mega",
        "kilo",
        "hecto",
        "deca",
        "deci",
        "centi",
        "milli",
        "micro",
        "nano",
        "pico",
        "femto",
        "atto",
        "zepto",
        "yocto",
        "ronto",
        "quecto",
    )
    for prefix in prefixes:
        name = prefix + "meter"
        assert ours.Quantity(1, name).m_as("meter") == pytest.approx(
            upstream.Quantity(1, name).m_as("meter")
        ), name


def test_all_base_dimensions_and_compatible_unit_queries(registries):
    ours, upstream = registries
    base_units = (
        "meter",
        "kilogram",
        "second",
        "ampere",
        "kelvin",
        "mole",
        "candela",
    )
    for unit in base_units:
        assert dims(ours(unit)) == dims(upstream(unit))
    compatible = {str(unit) for unit in ours.get_compatible_units("meter")}
    assert {"meter", "inch", "foot", "yard", "mile"} <= compatible
    assert ours.get_name("metres") == "meter"
    assert ours.get_symbol("meter") == "m"


def test_custom_definition_matches_pint():
    ours, upstream = mp.UnitRegistry(), pint.UnitRegistry()
    ours.define("smoot = 1.7018 * meter = smoots")
    upstream.define("smoot = 1.7018 * meter = smoots")
    assert ours.Quantity(3, "smoot").m_as("meter") == pytest.approx(
        upstream.Quantity(3, "smoot").m_as("meter")
    )
    assert ours.smoots == ours.smoot


def test_ito_and_m_as_match_pint(registries):
    ours, upstream = registries
    values = np.array([1.0, 2.0])
    a = ours.Quantity(values, "yard")
    b = upstream.Quantity(np.array([1.0, 2.0]), "yard")
    assert a.ito("meter") is None
    assert b.ito("meter") is None
    assert a.magnitude is values
    assert np.allclose(a.magnitude, b.magnitude)
    assert np.allclose(a.m_as("foot"), b.m_as("foot"))


def test_indexing_and_iteration_keep_units(registries):
    ours, upstream = registries
    a = ours.Quantity([1, 2, 3], "meter")
    b = upstream.Quantity([1, 2, 3], "meter")
    assert a[1].magnitude == b[1].magnitude
    assert [item.magnitude for item in a] == [item.magnitude for item in b]
    assert all(item.units == ours.meter for item in a)


def test_quantity_comparisons_and_unary_arithmetic_match_pint(registries):
    ours, upstream = registries
    a = ours.Quantity(np.array([1.0, 2.0, 3.0]), "meter")
    b = upstream.Quantity(np.array([1.0, 2.0, 3.0]), "meter")
    other_a = ours.Quantity(250, "centimeter")
    other_b = upstream.Quantity(250, "centimeter")
    assert np.array_equal(a < other_a, b < other_b)
    assert np.array_equal(a >= other_a, b >= other_b)
    assert np.array_equal(abs(-a).magnitude, abs(-b).magnitude)


def test_fractional_power_matches_pint(registries):
    ours, upstream = registries
    a = (ours.Quantity(9, "meter ** 2")) ** 0.5
    b = (upstream.Quantity(9, "meter ** 2")) ** 0.5
    assert a.magnitude == pytest.approx(b.magnitude)
    assert dims(a) == dims(b)


def test_get_base_units_matches_pint(registries):
    ours, upstream = registries
    factor_a, base_a = ours.get_base_units("mile/hour")
    factor_b, base_b = upstream.get_base_units("mile/hour")
    assert factor_a == pytest.approx(factor_b)
    assert dims(ours.Quantity(1, base_a)) == dims(upstream.Quantity(1, base_b))


def test_numpy_left_multiplication_creates_one_quantity(registries):
    ours, _ = registries
    quantity = np.arange(5.0) * ours.meter
    assert isinstance(quantity, mp.Quantity)
    assert np.array_equal(quantity.magnitude, np.arange(5.0))
    assert quantity.units == ours.meter

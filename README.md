# mojo-pint

`mojo-pint` is a standalone Mojo port of the unit parsing and dimensional
analysis core of [Pint](https://pint.readthedocs.io/). It provides Pint-shaped
`UnitRegistry`, `Unit`, and `Quantity` APIs in Python and sends bulk numeric
conversion work to a compiled Mojo shared library.

The aim is a useful covered subset, not a claim of complete Pint
compatibility. The test suite checks every bundled unit and decimal prefix
against Pint, plus representative parity tests for each API behavior listed
below.

## Covered subset

- Unit expressions with multiplication, division, integer or decimal powers,
  parentheses, implicit multiplication, `per`, `square`, and `cubic`
- SI base units, all named SI derived units, decimal prefixes from quecto to
  quetta, and common time, angle, US/imperial, volume, pressure, energy, power,
  and temperature units
- Unit aliases, symbols, plurals, compatible-unit queries, and simple runtime
  definitions such as `smoot = 1.7018 * meter`
- Seven-base-dimension analysis, fractional dimensions, compatibility checks,
  dimensional errors, and base-unit reduction
- Scalar and NumPy-array quantities; conversion, in-place conversion,
  arithmetic, powers, comparisons, indexing, and iteration
- Correct affine temperature-point conversions and temperature-difference
  arithmetic for Celsius, Fahrenheit, Kelvin, and Rankine

Not covered are Pint contexts, measurement uncertainty, logarithmic units,
NumPy ufunc dispatch, localization, unit systems beyond SI base reduction,
custom formatters, or loading Pint definition files. Runtime `define` supports
ordinary multiplicative definitions, not Pint's complete definition language.

## Install

The repository pins a Mojo nightly known to compile the FFI layer:

```bash
pixi install
pixi run build
```

The build creates `dist/libmojo-pint.so`. Tests and benchmarks are run with:

```bash
pixi run test
pixi run bench
```

## Usage

```python
import numpy as np
import mojopint as pint

ureg = pint.UnitRegistry()

acceleration = ureg("9.80665 m / s^2")
print(acceleration.to("ft / s^2"))

temperatures = ureg.Quantity(np.array([0.0, 20.0, 100.0]), "degC")
print(temperatures.to("degF"))

energy = 12 * ureg.newton * (3 * ureg.meter)
assert energy.to("joule").magnitude == 36
assert energy.check("[mass] * [length] ** 2 / [time] ** 2")
```

Run the example from the checkout with
`pixi run python examples/quickstart.py`.

## Benchmarks

Benchmark timings are filled from `pixi run bench`, which warms both
implementations, checks their results for parity, and reports the best of five
runs unless shown otherwise.

Measured by the command above on an Intel Xeon E5-2697 v4 at 2.30 GHz, Linux
6.8.0-136-generic, Python 3.13.14:

| Benchmark | Mojo-Pint | Pint | Pint / Mojo-Pint | Result |
|---|---:|---:|---:|:---|
| meter to inch, float64 array (5M) | 10.56 ms | 39.07 ms | 3.70x | faster |
| degC to degF, float64 array (5M) | 11.83 ms | 178.77 ms | 15.11x | faster |
| add meter + centimeter arrays (3M) | 10.21 ms | 15.96 ms | 1.56x | faster |
| parse cached compound expression (50k) | 190.35 ms | 11482.14 ms | 60.32x | faster |
| unit compatibility checks (50k) | 117.10 ms | 3625.00 ms | 30.96x | faster |

The parsing and compatibility cases use the best of three runs because each
Pint sample takes several seconds. These are end-to-end Python API timings,
including ctypes and allocation overhead, not isolated kernel timings.

## How it works

The Python layer owns registry state, tokenization, expression parsing, aliases,
prefix expansion, and Pint-compatible objects and exceptions. Parsed
expressions are cached. Each unit stores a scale and offset to SI plus a
fixed-width seven-element dimensional vector ordered as length, mass, time,
current, temperature, substance, and luminosity.

Large conversions use the affine operation `destination = source * scale +
shift` in a SIMD Mojo kernel with a scalar remainder loop. Small arrays stay
serial; large conversions are split into independent CPU tasks. NumPy provides
contiguous `float64` input and output arrays; ctypes passes their addresses
across the C ABI, and Mojo reconstructs mutable `UnsafePointer` values
internally. Python retains ownership of every buffer for the full call,
contiguous `float64` inputs cross the boundary without a copy, and in-place
conversion reuses writable storage. Empty arrays are handled before pointer
construction. Complex and extended-precision arrays stay in NumPy so the FFI
cannot silently narrow them.

There is no GPU path.

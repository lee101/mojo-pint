"""Benchmarks against Pint on identical inputs.

Run only through ``pixi run bench``; that task holds the shared machine lock.
"""

from __future__ import annotations

import math
import os
import platform
import sys
import time

import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python")
)

import mojopint as mp  # noqa: E402
import pint  # noqa: E402


def timeit(function, repeat=5):
    best = math.inf
    for _ in range(repeat):
        started = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - started)
    return best


def cpu_name():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


OURS = mp.UnitRegistry()
UPSTREAM = pint.UnitRegistry()
CASES = []


def case(name, repeat=5):
    def decorate(builder):
        CASES.append((name, repeat, builder))
        return builder

    return decorate


@case("meter to inch, float64 array (5M)")
def _():
    values = np.linspace(-1e5, 1e5, 5_000_000)
    ours = OURS.Quantity(values, "meter")
    upstream = UPSTREAM.Quantity(values, "meter")
    a = lambda: ours.to("inch")
    b = lambda: upstream.to("inch")
    assert np.allclose(a().magnitude, b().magnitude)
    return a, b


@case("degC to degF, float64 array (5M)")
def _():
    values = np.linspace(-273.15, 2000, 5_000_000)
    ours = OURS.Quantity(values, "degC")
    upstream = UPSTREAM.Quantity(values, "degC")
    a = lambda: ours.to("degF")
    b = lambda: upstream.to("degF")
    assert np.allclose(a().magnitude, b().magnitude)
    return a, b


@case("add meter + centimeter arrays (3M)")
def _():
    values = np.linspace(-1000, 1000, 3_000_000)
    ours_left = OURS.Quantity(values, "meter")
    ours_right = OURS.Quantity(values, "centimeter")
    pint_left = UPSTREAM.Quantity(values, "meter")
    pint_right = UPSTREAM.Quantity(values, "centimeter")
    a = lambda: ours_left + ours_right
    b = lambda: pint_left + pint_right
    assert np.allclose(a().magnitude, b().magnitude)
    return a, b


@case("parse cached compound expression (50k)", repeat=3)
def _():
    expression = "9.80665 kilogram * meter / second ** 2"

    def ours():
        for _ in range(50_000):
            OURS(expression)

    def upstream():
        for _ in range(50_000):
            UPSTREAM(expression)

    assert dims(OURS(expression)) == dims(UPSTREAM(expression))
    return ours, upstream


@case("unit compatibility checks (50k)", repeat=3)
def _():
    ours_left = OURS.Unit("newton * meter")
    pint_left = UPSTREAM.Unit("newton * meter")

    def ours():
        for _ in range(50_000):
            ours_left.is_compatible_with("kg * meter ** 2 / second ** 2")

    def upstream():
        for _ in range(50_000):
            pint_left.is_compatible_with("kg * meter ** 2 / second ** 2")

    assert ours_left.is_compatible_with(
        "kg * meter ** 2 / second ** 2"
    ) == pint_left.is_compatible_with("kg * meter ** 2 / second ** 2")
    return ours, upstream


def dims(quantity):
    return {name: float(power) for name, power in quantity.dimensionality.items()}


def main():
    print(f"Machine: {cpu_name()}; {platform.system()} {platform.release()}; Python {platform.python_version()}")
    print()
    print("| Benchmark | Mojo-Pint | Pint | Pint / Mojo-Pint | Result |")
    print("|---|---:|---:|---:|:---|")
    for name, repeat, builder in CASES:
        ours, upstream = builder()
        ours()
        upstream()
        ours_seconds = timeit(ours, repeat)
        pint_seconds = timeit(upstream, repeat)
        ratio = pint_seconds / ours_seconds
        result = "faster" if ratio >= 1 else "slower"
        print(
            f"| {name} | {ours_seconds * 1e3:.2f} ms | "
            f"{pint_seconds * 1e3:.2f} ms | {ratio:.2f}x | {result} |"
        )


if __name__ == "__main__":
    main()

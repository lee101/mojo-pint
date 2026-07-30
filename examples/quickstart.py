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

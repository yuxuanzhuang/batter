import math
import sys
import numpy

# Constants
delta_G = eval(sys.argv[1])  # kcal/mol
R = 1.987e-3  # kcal/(mol·K)
T = 310  # K

# Calculate Kd
Kd = math.exp(delta_G / (R * T))
print(f"Kd = {Kd:.2e} M")  # Output in scientific notation
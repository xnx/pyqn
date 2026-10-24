# constants.py
# Physical constants, in SI units, used to define units and conventions:
# the exact values of the SI defining constants and CODATA 2022 values of
# the others.
#
# Copyright (C) 2026 Christian Hill
# SPDX-License-Identifier: GPL-3.0-or-later

import math

# Exact: speed of light (m.s-1), Planck constant (J.s), elementary charge (C),
# Boltzmann constant (J.K-1) and Avogadro constant (mol-1).
c = 299792458.0
h = 6.62607015e-34
e = 1.602176634e-19
kB = 1.380649e-23
NA = 6.02214076e23
hbar = h / (2 * math.pi)

# CODATA 2022: atomic mass constant (kg), electron mass (kg), Hartree energy
# (J) and Bohr radius (m).
m_u = 1.66053906892e-27
m_e = 9.1093837139e-31
E_h = 4.3597447222060e-18
a0 = 5.29177210544e-11

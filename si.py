# -*- coding: utf-8 -*-

# Christian Hill
# v0.2 21/11/2012
# v0.1 28/11/2011
#
# Metadata relating to SI units and their prefixes


class SIPrefix(object):
    """ A little class describing SI prefixes. """
    def __init__(self, prefix, name, power):
        self.prefix = prefix
        self.name = name
        self.power = power
        self.fac = 10**power

# Here are the SI prefixes that we recognise.
si_prefixes = { u'y': SIPrefix(u'y', 'yocto', -24),
                u'z': SIPrefix(u'z', 'zepto', -21),
                u'a': SIPrefix(u'a', 'atto', -18),
                u'f': SIPrefix(u'f', 'femto', -15),
                u'p': SIPrefix(u'p', 'pico', -12),
                u'n': SIPrefix(u'n', 'nano', -9),
                u'μ': SIPrefix(u'μ', 'micro', -6),
                u'm': SIPrefix(u'm', 'milli', -3),
                u'c': SIPrefix(u'c', 'centi', -2),
                u'd': SIPrefix(u'd', 'deci', -1),
                u'k': SIPrefix(u'k', 'kilo', 3),
                u'M': SIPrefix(u'M', 'mega', 6),
                u'G': SIPrefix(u'G', 'giga', 9),
                u'T': SIPrefix(u'T', 'tera', 12),
                u'P': SIPrefix(u'P', 'peta', 15),
                u'E': SIPrefix(u'E', 'exa', 18),
                u'Z': SIPrefix(u'Z', 'zetta', 21),
                u'Y': SIPrefix(u'Y', 'yotta', 24),
              }

# The base SI unit stems for length, time, mass, amount of substance,
# thermodynamic temperature, luminous intenstiy and current respectively:
si_unit_stems = (u'm', u's', u'g', u'mol', u'K', u'cd', u'A')


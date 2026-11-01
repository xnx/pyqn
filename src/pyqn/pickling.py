# pickling.py
# Reading the pickles of the core objects, including those from before pyqn
# 2.1, when they were plain classes rather than frozen dataclasses.
#
# SPDX-License-Identifier: GPL-3.0-or-later

from dataclasses import fields


def reads_old_pickles(**missing):
    """
    Return a class decorator that lets a frozen dataclass read pickles from
    before pyqn 2.1. It must go above @dataclass, because in Python 3.10
    dataclass(frozen=True, slots=True) replaces any __setstate__ in the class.

    Their state is a dict of attributes, not a list of field values: attributes
    that are no longer fields are dropped, lists become tuples, and a field
    that is not in the dict takes its value from missing, or else its default.

    """

    def decorate(cls):
        def __setstate__(self, state):
            if not isinstance(state, dict):
                state = dict(zip((f.name for f in fields(self)), state))
            for f in fields(self):
                value = state.get(f.name, missing.get(f.name, f.default))
                if isinstance(value, list):
                    value = tuple(value)
                object.__setattr__(self, f.name, value)

        cls.__setstate__ = __setstate__
        return cls

    return decorate

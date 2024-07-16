# This file is part of tridiax, a toolkit for solving tridiagonal systems. tridiax is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from tridiax.__version__ import __version__  # noqa: F401

from tridiax.thomas import thomas_solve
from tridiax.divide_and_conquer import divide_conquer_solve, divide_conquer_index
from tridiax.stone import stone_solve

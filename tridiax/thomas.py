# This file is part of tridiax, a toolkit for solving tridiagonal systems. tridiax is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import jax.numpy as jnp
from jax import lax
from jax.numpy import ndarray


def thomas_solve(
    lower: ndarray,
    diag: ndarray,
    upper: ndarray,
    solve: ndarray,
    divide_last: bool = True,
) -> ndarray:
    """
    Return solution `x` to tridiagonal system `Ax = b` with Thomas algorithm.

    Args:
        divide_last: If `True`, the last row will be divided through by the diagonal
            such that the entire diagonal is 1. If `False`, this step will be skipped
            only for the very last value on the diagonal.
    """
    u, u_upper, y = thomas_triang_upper(
        lower, diag, upper, solve, divide_last=divide_last
    )
    return thomas_backsub_lower(y, u_upper, u)


def thomas_triang_lower(
    lower: jnp.ndarray,
    diag: jnp.ndarray,
    upper: jnp.ndarray,
    solve: jnp.ndarray,
    divide_last: bool = False,
):
    """
    Note that `divide_last=False` by default here, unlike in `thomas_solve`. This is to
    support this using `thomas_triang` in `Jaxley`.
    """
    n = len(solve)
    # Triangulation is only needed for matrizes, not scalars.
    if n > 1:
        u_upper = jnp.zeros(n - 1)
        u_upper = u_upper.at[0].set(upper[0] / diag[0])
        u_upper = lax.fori_loop(
            1, n - 1, _u_upper_update, (u_upper, upper, diag, lower)
        )[0]

        y = jnp.zeros(n)
        y = y.at[0].set(solve[0] / diag[0])
        y = lax.fori_loop(1, n - 1, _y_update, (solve, lower, y, diag, u_upper))[2]

        # Handle last row separately.
        newdiag = jnp.ones_like(diag)
        y = y.at[-1].set(solve[-1] - lower[-1] * y[-2])
        if divide_last:
            y = y.at[-1].set(y[-1] / (diag[-1] - lower[-1] * u_upper[-1]))
        else:
            newdiag = newdiag.at[-1].set(diag[-1] - lower[-1] * u_upper[-1])

        return newdiag, u_upper, y
    else:
        return diag, upper, solve


def thomas_triang_upper(
    lower: jnp.ndarray,
    diag: jnp.ndarray,
    upper: jnp.ndarray,
    solve: jnp.ndarray,
    divide_last: bool = False,
):
    """Triangulate system by removing the upper diagonal."""
    diag, lower, y = thomas_triang_lower(
        lower=jnp.flip(upper),
        diag=jnp.flip(diag),
        upper=jnp.flip(lower),
        solve=jnp.flip(solve),
        divide_last=divide_last,
    )
    return jnp.flip(diag), jnp.flip(lower), jnp.flip(y)


def thomas_backsub_upper(solve: jnp.ndarray, upper: jnp.ndarray, diag: jnp.ndarray):
    """Backsubstitute to remove the upper diagonal."""
    n = len(solve)
    x = jnp.zeros(n)
    x = x.at[0].set(solve[n - 1] / diag[n - 1])
    # This if-case should not be necessary, but even if fori does not do any
    # iterations, it still traces the body_fun and breaks without this if-case.
    if n > 1:
        x = lax.fori_loop(0, n - 1, _x_update, (solve, upper, x, n))[2]

    return jnp.flip(x)


def thomas_backsub_lower(solve: jnp.ndarray, lower: jnp.ndarray, diag: jnp.ndarray):
    """Backsubstitute to remove the lower diagonal."""
    solution = thomas_backsub_upper(jnp.flip(solve), jnp.flip(lower), jnp.flip(diag))
    return jnp.flip(solution)



def _u_upper_update(i, val):
    u_upper, upper, diag, lower = val
    u_upper = u_upper.at[i].set(upper[i] / (diag[i] - lower[i - 1] * u_upper[i - 1]))
    return (u_upper, upper, diag, lower)


def _y_update(i, val):
    solve, lower, y, diag, u_upper = val
    y = y.at[i].set(
        (solve[i] - lower[i - 1] * y[i - 1]) / (diag[i] - lower[i - 1] * u_upper[i - 1])
    )
    return (solve, lower, y, diag, u_upper)


def _x_update(i, val):
    y, u_upper, x, n = val
    x = x.at[i + 1].set(y[n - i - 2] - u_upper[n - i - 2] * x[i])
    return (y, u_upper, x, n)

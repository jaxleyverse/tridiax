from typing import Optional

import jax
import jax.numpy as jnp
from jax.numpy import ndarray
from jax import lax


def thomas_solve(
    lower: ndarray,
    diag: ndarray,
    upper: ndarray,
    solve: ndarray,
) -> ndarray:
    """
    Return solution `x` to tridiagonal system `Ax = b` with Thomas algorithm.
    """
    u_upper, y = thomas_triang(lower, diag, upper, solve)
    return thomas_backsub(y, u_upper)


def thomas_triang(
    lower: jnp.ndarray,
    diag: jnp.ndarray,
    upper: jnp.ndarray,
    solve: jnp.ndarray,
):
    n = len(solve)
    u_upper = jnp.zeros(n - 1)
    u_upper = u_upper.at[0].set(upper[0] / diag[0])
    u_upper = lax.fori_loop(1, n - 1, _w_update, (u_upper, upper, diag, lower))[0]

    y = jnp.zeros(n)
    y = y.at[0].set(solve[0] / diag[0])
    y = lax.fori_loop(1, n, _g_update, (solve, lower, y, diag, u_upper))[2]

    return u_upper, y


def thomas_backsub(
    solve: jnp.ndarray, upper: jnp.ndarray, diag: Optional[jnp.ndarray] = None
):
    n = len(solve)
    x = jnp.zeros(n)
    x = x.at[0].set(solve[n - 1])
    x = lax.fori_loop(0, n - 1, _p_update, (solve, upper, x, n))[2]

    return jnp.flip(x)


def _w_update(i, val):
    w, c, b, a = val  # c=upper, b=diag, a=lower
    w = w.at[i].set(c[i] / (b[i] - a[i - 1] * w[i - 1]))
    return (w, c, b, a)


def _g_update(i, val):
    d, a, g, b, w = val  # d=solve, a=lower, g=g, b=diag, w=w
    g = g.at[i].set((d[i] - a[i - 1] * g[i - 1]) / (b[i] - a[i - 1] * w[i - 1]))
    return (d, a, g, b, w)


def _p_update(i, val):
    g, w, p, n = val
    p = p.at[i + 1].set(g[n - i - 2] - w[n - i - 2] * p[i])
    return (g, w, p, n)

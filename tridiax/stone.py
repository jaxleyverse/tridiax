import jax
import jax.numpy as jnp
from jax.numpy import ndarray
from jax import lax


def stone_solve(
    lower: ndarray,
    diag: ndarray,
    upper: ndarray,
    solve: ndarray,
) -> ndarray:
    """
    Return solution `x` to tridiagonal system `Ax = b` with Stone's algorithm.
    """
    n = len(solve)
    w = jnp.zeros(n - 1)
    g = jnp.zeros(n)
    x = jnp.zeros(n)

    w = w.at[0].set(upper[0] / diag[0])
    g = g.at[0].set(solve[0] / diag[0])

    val = (w, upper, diag, lower)
    w = lax.fori_loop(1, n - 1, _w_update, val)[0]

    val = (solve, lower, g, diag, w)
    g = lax.fori_loop(1, n, _g_update, val)[2]

    x = x.at[0].set(g[n - 1])
    val = (g, w, x, n)
    x = lax.fori_loop(0, n - 1, _p_update, val)[2]

    return jnp.flip(x)


def _w_update(i, val):
    w, c, b, a = val
    w = w.at[i].set(c[i] / (b[i] - a[i - 1] * w[i - 1]))
    return (w, c, b, a)


def _g_update(i, val):
    d, a, g, b, w = val
    g = g.at[i].set((d[i] - a[i - 1] * g[i - 1]) / (b[i] - a[i - 1] * w[i - 1]))
    return (d, a, g, b, w)


def _p_update(i, val):
    g, w, p, n = val
    p = p.at[i + 1].set(g[n - i - 2] - w[n - i - 2] * p[i])
    return (g, w, p, n)

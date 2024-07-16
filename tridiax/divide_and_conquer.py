# This file is part of tridiax, a toolkit for solving tridiagonal systems. tridiax is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Optional

from math import log2
from jax import vmap
import jax.numpy as jnp
from jax.numpy import ndarray


def divide_conquer_solve(
    lower: ndarray,
    diag: ndarray,
    upper: ndarray,
    solve: ndarray,
    indexing: Optional[ndarray] = None,
) -> ndarray:
    """
    Runs the divide and conquer algorithm to solve the tridiagonal system.

    Args:
        indexing: After having solved the linear system, the solution has to be
            rearraged into the correct order. If `indexing=None`, the reordering is
            done automatically, but this can take a few seconds. Instead, if
            `indexing` is passed, it is direclty used to reorder the result. This can
            be helpful if `divide_and_conquer` is applied successively for many systems
            of the same size. Use ```indexing = divide_conquer_index(dim)``` to
            precompute the indizes.
    """
    dim = len(diag)

    lower = jnp.expand_dims(lower, axis=0)
    diag = jnp.expand_dims(diag, axis=0)
    upper = jnp.expand_dims(upper, axis=0)
    solve = jnp.expand_dims(solve, axis=0)

    # Reduce the system.
    for _ in range(int(log2(dim))):
        a_bars, b_bars, c_bars, solve_bars = vmap(_reduce_system)(
            (lower, diag, upper, solve)
        )
        upper = _split(c_bars)
        diag = _split(b_bars)
        lower = _split(a_bars)
        solve = _split(solve_bars)

    # Solve the (now diagonal) system.
    x = solve / diag

    if indexing is None:
        return jnp.squeeze(_reorder(x, dim))
    else:
        return jnp.squeeze(x)[indexing]


def divide_conquer_index(dim: int):
    return _reorder(jnp.expand_dims(jnp.arange(0, dim), 1), dim)[0]


def _reorder(x, dim):
    """
    Reorder the solution.
    """
    for _ in range(int(log2(dim))):
        x = _merge_dims(x)
        x = vmap(_rearrange_dims, 0)(x)
    return x


def _reduce_system(vecs):
    lower, diag, upper, solve = vecs
    a = upper
    b = diag
    c = lower
    r = solve

    alpha1 = -a / b[1:]
    beta1 = -c / b[:-1]

    a_bar = alpha1[:-1] * a[1:]
    c_bar = beta1[1:] * c[:-1]

    b_bar = b.at[:-1].set(b[:-1] + alpha1 * c)
    b_bar = b_bar.at[1:].set(b_bar[1:] + beta1 * a)

    r_bar = r.at[:-1].set(r[:-1] + alpha1 * r[1:])
    r_bar = r_bar.at[1:].set(r_bar[1:] + beta1 * r[:-1])
    return (c_bar, b_bar, a_bar, r_bar)


def _split(vector):
    r = jnp.asarray([vector[:, ::2], vector[:, 1::2]])
    r2 = jnp.transpose(r, (1, 0, 2))
    return jnp.concatenate(r2, axis=0)


def _merge_dims(x):
    return jnp.reshape(x, (x.shape[0] // 2, x.shape[1] * 2))


def _rearrange_dims(x):
    dim = len(x)

    x_even = x[: -dim // 2]
    x_odd = x[dim // 2 :]

    x = x.at[::2].set(x_even)
    x = x.at[1::2].set(x_odd)
    return x

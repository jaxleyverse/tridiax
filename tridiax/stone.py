# This file is part of tridiax, a toolkit for solving tridiagonal systems. tridiax is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Tuple

import jax.numpy as jnp


def stone_solve(
    lower: jnp.ndarray,
    diag: jnp.ndarray,
    upper: jnp.ndarray,
    solve: jnp.ndarray,
    stabilize: bool = True,
    optimized_lu: bool = True,
) -> jnp.ndarray:
    """
    Return solution `x` to tridiagonal system `Ax = b` with Stone's algorithm.

    Proceeds in three steps (just like Thomas algorithm):
    1) LU decomposition: LUx = b
    2) Triangularization: Solve Ly = b
    3) Backsubstitution: Solve Ux = y
    """
    u, upper, y = stone_triang(
        lower, diag, upper, solve, stabilize=stabilize, optimized_lu=optimized_lu
    )
    x = stone_backsub(y, upper, u)

    return x


def stone_triang(
    lower: jnp.ndarray,
    diag: jnp.ndarray,
    upper: jnp.ndarray,
    solve: jnp.ndarray,
    stabilize: bool = True,
    optimized_lu: bool = True,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    LU decomposition. `u` is the diagonal of U. `m` is the lower diagonal of `l`.
    The upper diagonal of U is `upper`. The main diagonal of L is 1. Notation follows
    Stone (1973).
    """
    if optimized_lu:
        u, m = _lu(lower, diag, upper, solve, stabilize=stabilize)
    else:
        u, m = _lu_matmul(lower, diag, upper, solve, stabilize=stabilize)
    y = _solve_l(solve, m)
    return u, upper, y


def stone_backsub(
    solve: jnp.ndarray, upper: jnp.ndarray, diag: jnp.ndarray
) -> jnp.ndarray:
    return _solve_u(solve, upper, diag)


def _lu_serial(
    lower: jnp.ndarray,
    diag: jnp.ndarray,
    upper: jnp.ndarray,
    solve: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    N = len(solve)
    ui = jnp.zeros_like(diag)
    ui = ui.at[0].set(diag[0])

    for i in range(1, N):
        ui = ui.at[i].set(diag[i] - lower[i - 1] * upper[i - 1] / ui[i - 1])

    mi = jnp.zeros_like(lower)
    mi = mi.at[0].set(lower[0] / diag[0])

    for i in range(1, N - 1):
        mi = mi.at[i].set(lower[i] / (diag[i] - upper[i - 1] * mi[i - 1]))

    return ui, mi


def _lu(
    lower: jnp.ndarray,
    diag: jnp.ndarray,
    upper: jnp.ndarray,
    solve: jnp.ndarray,
    stabilize: bool = True,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    N = len(solve)
    E = lower
    F = upper
    D = diag
    EF = jnp.concatenate([jnp.asarray([0.0]), -E * F])
    QIM2 = jnp.ones((N + 2))
    QIM1 = jnp.concatenate([jnp.asarray([1.0]), diag])
    QI = diag
    QI = QI.at[1:].set(diag[1:] * diag[:-1] + EF[1:])
    temp = jnp.zeros_like(diag)
    U = jnp.zeros_like(diag)
    M = jnp.zeros_like(upper)

    i = 2
    while i <= N / 2:
        if stabilize:
            av_qi = jnp.mean(QI)
            QIM2 /= av_qi
            QIM1 /= av_qi
            QI /= av_qi

        temp = temp.at[i - 2 : N].set(
            QIM1[i - 1 : N + 1] * QIM1[: N - i + 2]
            + EF[: N - i + 2] * QIM2[i : N + 2] * QIM2[: N - i + 2]
        )
        QIM1 = QIM1.at[i : N + 1].set(
            QI[i - 1 : N] * QIM1[: N - i + 1]
            + EF[: N - i + 1] * QIM1[i : N + 1] * QIM2[: N - i + 1]
        )
        QIM2 = QIM2.at[i : N + 2].set(temp[i - 2 : N])
        QI = QI.at[i:N].set(D[i:N] * QIM1[i:N] + EF[i:N] * QIM2[i:N])

        i += i

    U = U.at[:N].set(QI[:N] / QIM1[:N])
    M = M.at[: N - 1].set(E[: N - 1] / U[: N - 1])

    return U, M


def _lu_matmul(
    lower: jnp.ndarray,
    diag: jnp.ndarray,
    upper: jnp.ndarray,
    solve: jnp.ndarray,
    stabilize: bool = True,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Implements an alternative way for computing the LU decomposition."""
    N = len(solve)
    E = lower
    M = jnp.zeros_like(upper)

    a_vals = jnp.zeros((N - 1, 2, 2))
    a_vals = a_vals.at[:, 0, 0].set(diag[1:])
    a_vals = a_vals.at[:, 0, 1].set(-upper * lower)
    a_vals = a_vals.at[:, 1, 0].set(1.0)

    i = 1
    while i < N:
        product = jnp.einsum("bij, bjk -> bik", a_vals[i : N - 1], a_vals[: N - i - 1])
        a_vals = a_vals.at[i : N - 1].set(product)

        if stabilize:
            a_vals = jnp.einsum(
                "bij, b -> bij", a_vals, 1 / jnp.mean(a_vals, axis=(1, 2))
            )

        i += i

    q_init = jnp.asarray([1.0, diag[0]])
    qs = jnp.einsum("bij, j -> bi", a_vals, q_init)

    qs = jnp.concatenate([jnp.asarray([[diag[0], 1.0]]), qs])
    U = qs[:, 0] / qs[:, 1]

    M = M.at[: N - 1].set(E[: N - 1] / U[: N - 1])
    return U, M


def _solve_l(solve: jnp.ndarray, m: jnp.ndarray) -> jnp.ndarray:
    N = len(solve)
    Yi = solve
    Mi = -m
    i = 1

    while i <= N / 2:
        Yi = Yi.at[i:N].set(Yi[i:N] + Yi[: N - i] * Mi[i - 1 : N - 1])
        Mi = Mi.at[i : N - 1].set(Mi[i : N - 1] * Mi[: N - i - 1])
        i += i

    return Yi


def _solve_u(solve: jnp.ndarray, upper: jnp.ndarray, diag: jnp.ndarray) -> jnp.ndarray:
    N = len(diag)

    upper_diag_1 = jnp.asarray(upper / diag[:-1])
    yi_diag_1 = solve / diag

    Yi_2 = yi_diag_1
    Mi_2 = -upper_diag_1

    i = 1

    while i <= N / 2:
        Yi_2 = Yi_2.at[: N - i].set(Yi_2[: N - i] + Yi_2[i:N] * Mi_2[: N - i])
        Mi_2 = Mi_2.at[: N - i - 1].set(Mi_2[: N - i - 1] * Mi_2[i : N - 1])

        i += i

    return Yi_2

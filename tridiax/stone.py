from typing import Tuple
import jax
import jax.numpy as jnp
from jax import lax


def stone_solve(
    lower: jnp.ndarray, diag: jnp.ndarray, upper: jnp.ndarray, solve: jnp.ndarray
) -> jnp.ndarray:
    """
    Return solution `x` to tridiagonal system `Ax = b` with Stone's algorithm.

    Proceeds in three steps (just like Thomas algorithm):
    1) LU decomposition: LUx = b
    2) Triangularization: Solve Ly = b
    3) Backsubstitution: Solve Ux = y
    """

    # LU decomposition. `u` is the diagonal of U. `m` is the lower diagonal of `l`.
    # The upper diagonal of U is `upper`. The main diagonal of L is 1. Notation follows
    # Stone (1973).
    u, m = _lu(lower, diag, upper, solve)

    # Triangularization.
    y = _triang(solve, m)

    # Backsubstituion.
    x = _backsub(y, upper, u)

    return x


def _lu(
    lower: jnp.ndarray, diag: jnp.ndarray, upper: jnp.ndarray, solve: jnp.ndarray
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


def _triang(solve: jnp.ndarray, m: jnp.ndarray) -> jnp.ndarray:
    N = len(solve)
    Yi = solve
    Mi = -m
    i = 1

    while i <= N / 2:
        Yi = Yi.at[i:N].set(Yi[i:N] + Yi[: N - i] * Mi[i - 1 : N - 1])
        Mi = Mi.at[i : N - 1].set(Mi[i : N - 1] * Mi[: N - i - 1])
        i += i

    return Yi


def _backsub(solve: jnp.ndarray, upper: jnp.ndarray, diag: jnp.ndarray) -> jnp.ndarray:
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

import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".8"

from copy import deepcopy

import jax
import jax.numpy as jnp
import numpy as np
from tridiax import thomas_solve, divide_conquer_solve, divide_conquer_index
import pytest


device_str = "cpu"
jax.config.update("jax_platform_name", device_str)


@pytest.mark.parametrize("solve_fn", [thomas_solve, divide_conquer_solve])
def solver_api(solve_fn):
    dim = 1024
    _ = np.random.seed(0)
    diag = jnp.asarray(np.random.randn(dim))
    upper = jnp.asarray(np.random.randn(dim - 1))
    lower = jnp.asarray(np.random.randn(dim - 1))
    solve = jnp.asarray(np.random.randn(dim))
    solution = solve_fn(lower, diag, upper, solve)
    assert solution.shape == (dim,)


@pytest.mark.parametrize("solve_fn", [thomas_solve, divide_conquer_solve])
def test_solver_accuracy(solve_fn):
    dim = 32
    _ = np.random.seed(0)
    diag = jnp.asarray(np.random.randn(dim))
    upper = jnp.asarray(np.random.randn(dim - 1))
    lower = jnp.asarray(np.random.randn(dim - 1))
    solve = jnp.asarray(np.random.randn(dim))
    solution = solve_fn(lower, diag, upper, solve)

    tridiag_matrix = np.zeros((dim, dim))
    for row in range(dim):
        for col in range(dim):
            if row == col:
                tridiag_matrix[row, col] = deepcopy(diag[row])
            if row + 1 == col:
                tridiag_matrix[row, col] = deepcopy(upper[row])
            if row - 1 == col:
                tridiag_matrix[row, col] = deepcopy(lower[col])
    solution_np = np.linalg.solve(tridiag_matrix, solve)
    error = np.abs(solution - solution_np) / solution_np
    assert np.all(error < 1e-4)


def test_divide_and_conquer_preinit():
    dim = 1024
    _ = np.random.seed(0)
    diag = jnp.asarray(np.random.randn(dim))
    upper = jnp.asarray(np.random.randn(dim - 1))
    lower = jnp.asarray(np.random.randn(dim - 1))
    solve = jnp.asarray(np.random.randn(dim))

    indexing = divide_conquer_index(dim)
    solution_preinit = divide_conquer_solve(
        lower, diag, upper, solve, indexing=indexing
    )
    assert solution_preinit.shape == (dim,)
    solution_from_scratch = divide_conquer_solve(lower, diag, upper, solve)

    error = np.abs(solution_preinit - solution_from_scratch) / solution_preinit
    assert np.all(error < 1e-4)

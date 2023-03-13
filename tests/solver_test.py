import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".8"

from copy import deepcopy

import jax
import jax.numpy as jnp
from jax import jit, grad
import numpy as np
from tridiax import (
    thomas_solve,
    divide_conquer_solve,
    stone_solve,
    divide_conquer_index,
)
import pytest


device_str = "cpu"
jax.config.update("jax_platform_name", device_str)


@pytest.mark.parametrize("solve_fn", [thomas_solve, divide_conquer_solve, stone_solve])
def test_solver_api(solve_fn):
    dim = 1024
    _ = np.random.seed(0)
    diag = jnp.asarray(np.random.randn(dim))
    upper = jnp.asarray(np.random.randn(dim - 1))
    lower = jnp.asarray(np.random.randn(dim - 1))
    solve = jnp.asarray(np.random.randn(dim))
    solution = solve_fn(lower, diag, upper, solve)
    assert solution.shape == (dim,)


@pytest.mark.parametrize("solve_fn", [thomas_solve, divide_conquer_solve, stone_solve])
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


@pytest.mark.parametrize("solve_fn", [thomas_solve, divide_conquer_solve, stone_solve])
def test_jit(solve_fn):
    dim = 32
    _ = np.random.seed(0)
    diag = jnp.asarray(np.random.randn(dim))
    upper = jnp.asarray(np.random.randn(dim - 1))
    lower = jnp.asarray(np.random.randn(dim - 1))
    solve = jnp.asarray(np.random.randn(dim))

    jitted_solver = jit(solve_fn)
    _ = jitted_solver(lower, diag, upper, solve)


@pytest.mark.parametrize("solve_fn", [thomas_solve, divide_conquer_solve, stone_solve])
def test_grad(solve_fn):
    dim = 32
    _ = np.random.seed(0)
    diag = jnp.asarray(np.random.randn(dim))
    upper = jnp.asarray(np.random.randn(dim - 1))
    lower = jnp.asarray(np.random.randn(dim - 1))
    solve = jnp.asarray(np.random.randn(dim))

    def sum_solution(vals):
        lower, diag, upper, solve = vals
        x = solve_fn(lower, diag, upper, solve)
        return jnp.sum(x)

    jitted_grad = jit(grad(sum_solution))
    gradient = jitted_grad((lower, diag, upper, solve))
    for g in gradient:
        assert jnp.invert(jnp.any(jnp.isnan(g))), "Found NaN in gradient."


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

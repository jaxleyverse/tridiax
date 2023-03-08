# tridiax
Solvers for tridiagonal systems in jax (used by Michael for multicompartment models).

### Implemented solvers

- Thomas algorithm
- [Divide and conquer](https://courses.engr.illinois.edu/cs554/fa2013/notes/09_tridiagonal_8up.pdf)

### Usage

```python
from tridiax import thomas_solve, divide_conquer_solve

dim = 1024
diag = jnp.asarray(np.random.randn(dim))
upper = jnp.asarray(np.random.randn(dim - 1))
lower = jnp.asarray(np.random.randn(dim - 1))
solve = jnp.asarray(np.random.randn(dim))
solution = thomas_solve(lower, diag, upper, solve)
```

### Optimization

If many systems of the same size are solved and the divide and conquer algorithm is used, it helps to precompute the reordering indizes:
```python
from tridiax import divide_conquer_solve, divide_conquer_index

dim = 1024
diag = jnp.asarray(np.random.randn(dim))
upper = jnp.asarray(np.random.randn(dim - 1))
lower = jnp.asarray(np.random.randn(dim - 1))
solve = jnp.asarray(np.random.randn(dim))

indexing = divide_conquer_index(dim)
solution = thomas_solve(lower, diag, upper, solve, indexing=indexing)
```

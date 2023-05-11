[![Tests](https://github.com/mackelab/sbi/workflows/Tests/badge.svg?branch=main)](https://github.com/mackelab/sbi/actions)

# tridiax
`tridiax` implements solvers for tridiagonal systems in jax. All solvers support CPU and GPU, are compatible with `jit` compilation and can be differentiated with `grad`.


### Implemented solvers

- [Thomas algorithm](http://www.industrial-maths.com/ms6021_thomas.pdf)
- [Divide and conquer](https://courses.engr.illinois.edu/cs554/fa2013/notes/09_tridiagonal_8up.pdf)
- [Stone's algorithm](https://dl.acm.org/doi/pdf/10.1145/321738.321741)

Generally, Thomas algorithm will be faster on CPU whereas the divide and conquer
algorithm and Stone's algorithm will be faster on GPU.


### Known limitations

Currently, all solvers are only tested for systems whose dimensionality is an exponential of `2`.


### Usage

```python
from tridiax import thomas_solve, divide_conquer_solve, stone_solve

dim = 1024
diag = jnp.asarray(np.random.randn(dim))
upper = jnp.asarray(np.random.randn(dim - 1))
lower = jnp.asarray(np.random.randn(dim - 1))
solve = jnp.asarray(np.random.randn(dim))
solution = thomas_solve(lower, diag, upper, solve)
```

If many systems of the same size are solved and the divide and conquer algorithm is used, it helps to precompute the reordering indizes:
```python
from tridiax import divide_conquer_solve, divide_conquer_index

dim = 1024
diag = jnp.asarray(np.random.randn(dim))
upper = jnp.asarray(np.random.randn(dim - 1))
lower = jnp.asarray(np.random.randn(dim - 1))
solve = jnp.asarray(np.random.randn(dim))

indexing = divide_conquer_index(dim)
solution = divide_conquer_solve(lower, diag, upper, solve, indexing=indexing)
```

### Installation

```sh
git clone https://github.com/mackelab/tridiax.git
cd tridiax
pip install -e .
```

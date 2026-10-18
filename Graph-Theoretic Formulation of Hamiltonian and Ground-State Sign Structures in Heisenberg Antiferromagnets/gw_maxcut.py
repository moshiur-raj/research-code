import jax
import jax.numpy as jnp
from functools import partial
import optax

def _solve(
        w: jnp.ndarray,
        i: jnp.ndarray,
        j: jnp.ndarray,
        n: int,
        k: int,
        n_hyperplanes: int,
        n_iter: int,
        optimizer: optax.GradientTransformationExtraArgs,
        seed: int,
        dtype: jax.typing.DTypeLike,
        ) -> jnp.ndarray:

    ww = jnp.concatenate([w, w])[:, None]
    src = jnp.concatenate([i, j])
    dst = jnp.concatenate([j, i])

    def grad(V: jnp.ndarray) -> jnp.ndarray:
        G = jnp.zeros_like(V).at[src].add(ww * V[dst])
        # discard radial part of gradient, since we only want unit vectors in a hypersphere
        G -= (G * V).sum(axis=1, keepdims=True) * V

        return G

    def iterate(
            carry: tuple[jnp.ndarray, optax.OptState],
            x: None,
            ) -> tuple[tuple[jnp.ndarray, optax.OptState], None]:

        V, opt_state = carry
        updates, opt_state = optimizer.update(grad(V), opt_state)
        V = optax.apply_updates(V, updates)
        V /= jnp.linalg.norm(V, axis=1, keepdims=True)

        return (V, opt_state), None

    key_v, key_r = jax.random.split(jax.random.key(seed))
    V = jax.random.normal(key_v, (n, k), dtype=dtype)
    V /= jnp.linalg.norm(V, axis=1, keepdims=True)
    (V, _), _ = jax.lax.scan(iterate, init=(V, optimizer.init(V)), length=n_iter)

    R = jax.random.normal(key_r, (k, n_hyperplanes), dtype=dtype)
    S = (V @ R > 0).astype(dtype)
    S = 2 * S - 1
    cuts = (w[:, None] * S[i] * S[j]).sum(axis=0)

    return S[:, cuts.argmin()]

@jax.jit(static_argnames=("n", "k", "n_hyperplanes", "n_iter", "optimizer", "seed", "dtype"))
def gw_maxcut(
        w: jax.typing.ArrayLike,
        i: jax.typing.ArrayLike,
        j: jax.typing.ArrayLike,
        n: int,
        k: int,
        n_hyperplanes: int,
        n_iter: int,
        optimizer: optax.GradientTransformationExtraArgs,
        seed: int = 0,
        dtype: jax.typing.DTypeLike = jnp.float32,
        ) -> jnp.ndarray:

    w = jnp.asarray(w)
    i = jnp.asarray(i)
    j = jnp.asarray(j)

    w = (w / jnp.abs(w).max()).astype(dtype)

    if not n:
        n = int(jnp.maximum(i.max(), j.max())) + 1
    if not k:
        k = int(jnp.ceil((2*n)**0.5))

    return _solve(w, i, j, n, k, n_hyperplanes, n_iter, optimizer, seed, dtype)

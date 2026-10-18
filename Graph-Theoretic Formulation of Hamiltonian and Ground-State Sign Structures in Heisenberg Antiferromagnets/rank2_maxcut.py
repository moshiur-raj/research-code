from functools import partial
import jax
import jax.numpy as jnp
import optax

def _solve(
        w: jnp.ndarray,
        i: jnp.ndarray,
        j: jnp.ndarray,
        n: int,
        deterministic_cut: bool,
        n_iter: int,
        optimizer: optax.GradientTransformationExtraArgs,
        seed: int,
        dtype: jax.typing.DTypeLike = jnp.float32,
        ) -> jnp.ndarray:

    ww = jnp.concatenate([w, w])
    src = jnp.concatenate([i, j])
    dst = jnp.concatenate([j, i])

    def grad(theta: jnp.ndarray) -> jnp.ndarray:
        G = jnp.zeros_like(theta).at[src].add(ww * jnp.sin(theta[dst] - theta[src]))
        return G

    def iterate(
            carry: tuple[jnp.ndarray, optax.OptState],
            x: None,
            ) -> tuple[tuple[jnp.ndarray, optax.OptState], None]:

        theta, opt_state = carry
        updates, opt_state = optimizer.update(grad(theta), opt_state)
        theta = optax.apply_updates(theta, updates)

        return (theta, opt_state), None

    key = jax.random.key(seed)
    theta = jax.random.uniform(key, (n,), maxval=jnp.pi, dtype=dtype)
    (theta, _), _ = jax.lax.scan(iterate, init=(theta, optimizer.init(theta)), length=n_iter)

    if deterministic_cut:
        theta = theta % (2 * jnp.pi)
        alpha = jnp.append(jnp.sort(theta % jnp.pi), jnp.pi)
        alpha = (alpha[:-1] + alpha[1:]) / 2
        alpha = alpha[:, None]
        S = jnp.logical_or(theta <= alpha, theta > alpha + jnp.pi).astype(dtype)
        S = 2 * S - 1
        cuts = (w * S[:, i] * S[:, j]).sum(axis=-1)

        return S[cuts.argmin()]

    return theta

@jax.jit(static_argnames=("n", "deterministic_cut", "n_iter", "optimizer", "seed", "dtype"))
def rank2_maxcut(
        w: jax.typing.ArrayLike,
        i: jax.typing.ArrayLike,
        j: jax.typing.ArrayLike,
        n: int,
        n_iter: int,
        optimizer: optax.GradientTransformationExtraArgs,
        deterministic_cut: bool = True,
        seed: int = 0,
        dtype: jax.typing.DTypeLike = jnp.float32,
        ) -> jnp.ndarray:

    w = jnp.asarray(w)
    i = jnp.asarray(i)
    j = jnp.asarray(j)

    w = (w / jnp.abs(w).max()).astype(dtype)

    if not n:
        n = int(jnp.maximum(i.max(), j.max())) + 1

    return _solve(w, i, j, n, deterministic_cut, n_iter, optimizer, seed, dtype)

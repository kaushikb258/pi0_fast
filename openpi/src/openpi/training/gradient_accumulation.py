"""Sequential equal-sized microbatches; optimizer updates remain outside this helper."""
import jax
import jax.numpy as jnp


def mean_microbatch_gradients(loss_and_grad, rng, batch, steps):
    """Average losses/gradients at fixed weights, using distinct RNGs per microbatch.

    batch has the effective batch dimension. Only one microbatch's activations
    are live at a time. None leaves (optional observation fields) are preserved.
    """
    if steps < 1:
        raise ValueError("gradient accumulation steps must be positive")
    leaves = jax.tree.leaves(batch)
    size = leaves[0].shape[0]
    if size % steps or any(x.shape[0] != size for x in leaves):
        raise ValueError("batch leaves must share a batch size divisible by accumulation steps")
    if steps == 1:
        return loss_and_grad(rng, batch)
    micro_size = size // steps

    def evaluate(i):
        microbatch = jax.tree.map(
            lambda x: jax.lax.dynamic_slice_in_dim(x, i * micro_size, micro_size, axis=0), batch
        )
        return loss_and_grad(jax.random.fold_in(rng, i), microbatch)

    loss, grads = evaluate(0)
    # Accumulate in float32 even if a trainable parameter has reduced precision.
    grads = jax.tree.map(lambda x: x.astype(jnp.float32), grads)

    def add(i, carry):
        loss_sum, grad_sum = carry
        micro_loss, micro_grads = evaluate(i)
        return loss_sum + micro_loss, jax.tree.map(
            lambda a, b: a + b.astype(jnp.float32), grad_sum, micro_grads
        )

    loss, grads = jax.lax.fori_loop(1, steps, add, (loss, grads))
    return loss / steps, jax.tree.map(lambda x: x / steps, grads)

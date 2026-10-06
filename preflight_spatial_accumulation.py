"""One real forward/backward diagnostic. Never calls an optimizer update or saves weights."""
import argparse
import dataclasses
import json
import pathlib
import subprocess
import sys
import threading
import time

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

sys.path.insert(0, '/home/kb/pi0_fast/openpi')
from scripts.train import init_train_state
from openpi.training import config, data_loader, sharding

ROOT = pathlib.Path('/home/kb/pi0_fast')
OUT = ROOT / 'setup_logs/preflight_spatial_accumulation'
OUT.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--microbatch', type=int, choices=[1, 2], default=2)
    args = parser.parse_args()
    accumulation_steps = 8 // args.microbatch
    cfg = dataclasses.replace(config.get_config('pi0_fast_libero_spatial_lora'), batch_size=8, gradient_accumulation_steps=accumulation_steps)
    jax.config.update('jax_compilation_cache_dir', str(ROOT / 'cache/jax'))
    assert jax.default_backend() == 'gpu'
    assert cfg.batch_size == 8 and cfg.ema_decay is None
    memory = []
    stop = threading.Event()

    def monitor():
        while not stop.is_set():
            p = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], capture_output=True, text=True)
            if p.returncode == 0:
                memory.append(int(p.stdout.splitlines()[0]))
            stop.wait(0.5)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        mesh = sharding.make_mesh(cfg.fsdp_devices)
        batch_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
        loader = data_loader.create_data_loader(cfg, sharding=batch_sharding, shuffle=False, num_batches=1)
        obs, actions = next(iter(loader))
        assert tuple(actions.shape) == (8, 10, 7)
        for x in jax.tree.leaves((obs, actions)):
            assert np.isfinite(np.asarray(x)).all()
        active_tokens = int(np.asarray(obs.tokenized_prompt_mask)[0].sum())
        loss_tokens = int(np.asarray(obs.token_loss_mask)[0].sum())
        assert 0 < loss_tokens < active_tokens < cfg.model.max_token_len
        print(f'REAL_BATCH_PASS active_tokens={active_tokens} supervised_tokens={loss_tokens}', flush=True)
        rng = jax.random.key(cfg.seed)
        state, _ = init_train_state(cfg, rng, mesh, resume=False)
        jax.block_until_ready(state)
        trained = state.params.filter(cfg.trainable_filter)
        parameter_counts = {}
        for path, leaf in jax.tree_util.tree_leaves_with_path(trained):
            name = jax.tree_util.keystr(path)
            group = 'lora' if 'lora' in name else 'vision' if 'img' in name else 'other'
            parameter_counts[group] = parameter_counts.get(group, 0) + leaf.size
        print(f'INITIALIZED step={state.step}; trainable_parameters={parameter_counts}', flush=True)

        def evaluate(params, observation, target_actions):
            from openpi.training.gradient_accumulation import mean_microbatch_gradients
            def micro_loss_and_grad(micro_rng, microbatch):
                model = nnx.merge(state.model_def, params)
                model.train()
                def loss_fn(m):
                    return jnp.mean(m.compute_loss(micro_rng, *microbatch, train=True))
                return nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, cfg.trainable_filter))(model)
            return mean_microbatch_gradients(micro_loss_and_grad, rng, (observation, target_actions), accumulation_steps)

        started = time.monotonic()
        with sharding.set_mesh(mesh):
            # Direct JIT matches upstream training and avoids ArgInfo dataclass type checks.
            compiled = jax.jit(evaluate)
            loss, gradients = compiled(state.params, obs, actions)
            jax.block_until_ready((loss, gradients))
        elapsed = time.monotonic() - started
        # Inspect all gradient leaves on GPU. No tx.update, apply_updates, or training loop.
        finite = all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree.leaves(gradients))
        gradient_norm = float(optax.global_norm(gradients))
        lora_leaves = [leaf for path, leaf in jax.tree_util.tree_leaves_with_path(gradients) if 'lora' in jax.tree_util.keystr(path)]
        lora_norm = float(optax.global_norm(lora_leaves))
        assert np.isfinite(float(loss)) and finite and gradient_norm > 0 and lora_norm > 0
        assert int(state.step) == 0
        result = {
            'status': 'SPATIAL_ACCUMULATION_PREFLIGHT_PASS', 'config': cfg.name,
            'device': str(jax.devices()[0]), 'batch_size': cfg.batch_size,
            'microbatch_size': args.microbatch, 'accumulation_steps': accumulation_steps,
            'actions_shape': list(actions.shape),
            'image_shapes': {k:list(v.shape) for k,v in obs.images.items()},
            'active_tokens': active_tokens, 'supervised_tokens': loss_tokens,
            'max_token_len': cfg.model.max_token_len,
            'trainable_parameters': parameter_counts,
            'loss': float(loss), 'all_gradients_finite': finite,
            'gradient_norm': gradient_norm, 'lora_gradient_norm': lora_norm,
            'compile_and_backward_seconds': elapsed,
            'sampled_peak_gpu_memory_mib_including_other_processes': max(memory),
            'optimizer_updates': 0, 'step': int(state.step),
            'weights_saved': False, 'training_started': False,
        }
        (OUT / 'result.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2), flush=True)
    finally:
        stop.set()
        thread.join()

if __name__ == '__main__':
    main()

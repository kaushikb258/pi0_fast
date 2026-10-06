"""One batch-one base-model inference; synthetic normalized input with native base-model dimensions. No training."""
import json, pathlib, time
import jax
import jax.numpy as jnp
import numpy as np
import flax.nnx as nnx
from openpi.models import model as model_lib
from openpi.models.tokenizer import FASTTokenizer
from openpi.models.pi0_fast import Pi0FASTConfig
from openpi.policies.libero_policy import LiberoInputs
from openpi import transforms
from openpi.shared import nnx_utils

out = pathlib.Path('/home/kb/pi0_fast/setup_logs')
checkpoint = pathlib.Path('/home/kb/pi0_fast/cache/openpi/openpi-assets/checkpoints/pi0_fast_base')
assert jax.default_backend() == 'gpu', jax.devices()
print('GPU:', jax.devices(), flush=True)
cfg = Pi0FASTConfig()
start = time.monotonic()
params = model_lib.restore_params(checkpoint / 'params', dtype=jnp.bfloat16)
leaves = jax.tree.leaves(params)
assert all(x.devices() and all(d.platform == 'gpu' for d in x.devices()) for x in leaves)
num_params = sum(x.size for x in leaves)
print('Loaded pretrained parameters:', num_params, 'on GPU', flush=True)
model = cfg.load(params)
del params, leaves
print('Model restored in', time.monotonic()-start, 'seconds', flush=True)
tokenizer = FASTTokenizer(max_len=cfg.max_token_len, fast_tokenizer_path='/home/kb/pi0_fast/fast-tokenizer')
example = {
    'observation/image': np.zeros((224,224,3), dtype=np.uint8),
    'observation/wrist_image': np.zeros((224,224,3), dtype=np.uint8),
    'observation/state': np.zeros(cfg.action_dim, dtype=np.float32),
    'prompt': 'pick up the black bowl',
}
inputs = LiberoInputs(cfg.model_type)(example)
inputs = transforms.TokenizeFASTInputs(tokenizer)(inputs)
inputs = jax.tree.map(lambda x: jnp.asarray(x)[None,...], inputs)
obs = model_lib.Observation.from_dict(inputs)
sample = nnx_utils.module_jit(model.sample_actions)
start = time.monotonic()
print('Compiling and executing one autoregressive inference (up to 256 tokens)...', flush=True)
tokens = sample(jax.random.key(0), obs).block_until_ready()
seconds = time.monotonic()-start
assert all(d.platform == 'gpu' for d in tokens.devices())
raw = np.asarray(tokens[0], dtype=np.int32)
assert raw.size and np.any(raw != 0)
text = tokenizer._paligemma_tokenizer.decode(raw.tolist())
actions = tokenizer.extract_actions(raw, cfg.action_horizon, cfg.action_dim)
assert actions.shape == (cfg.action_horizon,cfg.action_dim) and np.isfinite(actions).all()
# The official decoder returns zeros if no Action prefix is present; report that explicitly.
action_prefix = 'Action: ' in text
valid_action_decode = False
if action_prefix:
    pg = tokenizer._paligemma_tokenizer.encode(text.split('Action: ')[1].split('|')[0].strip())
    fast_ids = tokenizer._act_tokens_to_paligemma_tokens(np.asarray(pg)).tolist()
    dct_string = tokenizer._fast_tokenizer.bpe_tokenizer.decode(fast_ids)
    valid_action_decode = len(dct_string) == cfg.action_horizon * cfg.action_dim
assert valid_action_decode, 'Decoded action coefficient count does not match the native base dimensions'
assert np.any(actions != 0), 'Unexpected all-zero action fallback'
np.savez(out / 'smoke_outputs.npz', tokens=raw, actions=actions)
result = {
    'status': 'GPU_INFERENCE_PASS', 'gpu': str(jax.devices()[0]), 'gpu_name': jax.devices()[0].device_kind,
    'checkpoint': 'gs://openpi-assets/checkpoints/pi0_fast_base',
    'config': 'Pi0FASTConfig native base defaults (32 actions x 32 dimensions)', 'parameters': num_params,
    'token_shape': list(tokens.shape), 'token_device': str(tokens.devices()),
    'action_shape': list(actions.shape), 'finite_actions': bool(np.isfinite(actions).all()),
    'action_prefix_present': action_prefix, 'valid_action_decode': valid_action_decode, 'decoded_text': text,
    'compile_and_inference_seconds': seconds,
    'note': 'Synthetic normalized input; no LIBERO normalization, fine-tuning, or task-quality evaluation.'
}
(out / 'smoke_result.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2), flush=True)

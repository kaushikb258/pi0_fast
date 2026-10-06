"""LIBERO Spatial residual adapter; legacy MimicGen feature hashes remain unchanged."""
import dataclasses
import hashlib
from pathlib import Path
from residual_policy import FrozenFeaturePolicy, ResidualPolicy, feature_code_hash as legacy_hash
from openpi.policies.policy_config import create_trained_policy
from openpi.shared.nnx_utils import module_jit
from openpi.training.config import get_config
from serve_audited_policy import AuditedTokenizer

PROJECT = Path('/home/kb/pi0_fast')
CONFIG = 'pi0_fast_libero_spatial_lora'
LAYERS = [4, 8, 12, 18]
DEFAULT_CHECKPOINT = PROJECT / 'checkpoints/pi0_fast_libero_spatial_lora/spatial_fast_plus_25k_b8_20260919_163905/24999'

def feature_code_hash():
    h = hashlib.sha256(legacy_hash().encode())
    for name in ['libero_residual_policy.py', 'libero_residual_data.py']:
        h.update((PROJECT / name).read_bytes())
    h.update(CONFIG.encode())
    return h.hexdigest()

class FrozenLiberoFeaturePolicy(FrozenFeaturePolicy):
    def __init__(self, checkpoint=DEFAULT_CHECKPOINT):
        self.checkpoint = str(Path(checkpoint).resolve())
        cfg = get_config(CONFIG)
        cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, fast_model_tokenizer=AuditedTokenizer))
        self.policy = create_trained_policy(cfg, self.checkpoint)
        self.features_jit = module_jit(self.policy._model.extract_prefix_features)
        self.feature_code_hash = feature_code_hash()

def create_libero_residual_policy(checkpoint, controller_path, beta=1.):
    import torch
    bundle = torch.load(controller_path, map_location='cpu', weights_only=True)
    if bundle.get('config') != CONFIG or bundle.get('layers') != LAYERS:
        raise ValueError('Expected a LIBERO Spatial controller with layers 4,8,12,18')
    if bundle['architecture']['num_layers'] != 4:
        raise ValueError('Controller architecture/layer mismatch')
    policy = ResidualPolicy(FrozenLiberoFeaturePolicy(checkpoint), controller_path, beta)
    policy.metadata.update(config=CONFIG, layers=LAYERS)
    return policy

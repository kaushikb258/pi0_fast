"""Frozen JAX VLA features and optional CPU residual head for inference."""
import dataclasses,hashlib,json
from pathlib import Path
import jax,jax.numpy as jnp,numpy as np
from openpi.models.model import Observation
from openpi.policies.policy_config import create_trained_policy
from openpi.shared.nnx_utils import module_jit
from openpi.training.config import get_config
from serve_audited_policy import AuditedTokenizer,last_decode

CONFIG='pi0_fast_mimicgen_square_lora'
LAYERS=[4,8,12,18]
PROJECT=Path('/home/kb/pi0_fast')
DEFAULT_CHECKPOINT=PROJECT/'checkpoints'/CONFIG/'mimicgen_square_fast_plus_continue_60k_b8_20260920_190251/50000'

def feature_code_hash():
    digest=hashlib.sha256()
    for name in ['openpi/src/openpi/models/gemma_fast.py','openpi/src/openpi/models/pi0_fast.py','residual_policy.py']:
        digest.update((PROJECT/name).read_bytes())
    return digest.hexdigest()

def checked_prefix_features(values):
    # Gemma bfloat16 activations can exceed float16's finite range (65504).
    features=np.array(values,dtype=np.float32,copy=True)
    if features.shape!=(4,2048):
        raise ValueError(f'Unexpected prefix feature shape: {features.shape}')
    if not np.isfinite(features).all():
        raise FloatingPointError('VLA prefix features contain nonfinite values before cache storage')
    return features

class FrozenFeaturePolicy:
    def __init__(self,checkpoint=DEFAULT_CHECKPOINT):
        self.checkpoint=str(Path(checkpoint).resolve())
        cfg=get_config(CONFIG)
        cfg=dataclasses.replace(cfg,model=dataclasses.replace(cfg.model,fast_model_tokenizer=AuditedTokenizer))
        self.policy=create_trained_policy(cfg,self.checkpoint)
        self.features_jit=module_jit(self.policy._model.extract_prefix_features)
        self.feature_code_hash=feature_code_hash()

    def infer_with_features(self,observation):
        # Never accept demonstrated actions into the frozen feature-extraction path.
        if 'actions' in observation:raise ValueError('Feature inputs must not include target actions')
        required=['observation/image','observation/wrist_image','observation/state','prompt']
        clean={k:observation[k] for k in required}
        x=self.policy._input_transform(jax.tree.map(lambda v:v,clean))
        x=jax.tree.map(lambda v:jnp.asarray(v)[None,...],x)
        features=checked_prefix_features(self.features_jit(Observation.from_dict(x))[0])
        output=self.policy.infer(clean)
        output['action_decode']=dict(last_decode)
        return output,features

class ResidualPolicy:
    def __init__(self,frozen,controller_path,beta=1.):
        import torch
        from residual_controller import ResidualMLP
        torch.set_num_threads(2)
        self.frozen=frozen;self.beta=float(beta)
        if not np.isfinite(self.beta) or not 0<=self.beta<=1:raise ValueError('beta must be between0 and1')
        self.path=str(Path(controller_path).resolve())
        bundle=torch.load(self.path,map_location='cpu',weights_only=True)
        if bundle['base_checkpoint']!=frozen.checkpoint:raise ValueError('Controller/base checkpoint mismatch')
        if bundle['feature_code_hash']!=frozen.feature_code_hash:raise ValueError('Controller feature extraction code changed')
        self.model=ResidualMLP(**bundle['architecture']);self.model.load_state_dict(bundle['state_dict'])
        self.model.eval();self.scales=np.array(bundle['motion_scales'],dtype=np.float32)
        self.metadata={'controller':self.path,'beta':self.beta,'layers':LAYERS,
            'layer_weights':self.model.layer_weights().detach().tolist(),'training_manifest':bundle['cache_manifest_hash']}

    def infer(self,observation):
        import torch
        output,features=self.frozen.infer_with_features(observation)
        base=np.asarray(output['actions']).copy()
        valid=bool(output['action_decode']['valid'])
        if valid and self.beta:
            with torch.inference_mode():
                correction=self.model(torch.from_numpy(features)[None])[0].numpy()*self.scales*self.beta
            output['actions']=base.copy();output['actions'][:,:6]+=correction
        output['residual_controller']={'applied':bool(valid and self.beta),'beta':self.beta}
        np.testing.assert_array_equal(output['actions'][:,6],base[:,6])
        return output

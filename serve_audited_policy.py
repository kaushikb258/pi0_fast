"""Official policy serving plus action-format telemetry; decoding behavior unchanged."""
import argparse
import dataclasses
import logging
import numpy as np
import jax
from openpi.models.tokenizer import FASTTokenizer
from openpi.policies.policy_config import create_trained_policy
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from openpi.training.config import get_config

from fast_decode_validation import validate_fast_tokens, DECODE_VALIDATION_VERSION

last_decode = {}
class AuditedTokenizer(FASTTokenizer):
    def extract_actions(self, tokens, action_horizon, action_dim):
        last_decode.clear()
        last_decode.update(valid=False, coefficient_count=-1, validation=DECODE_VALIDATION_VERSION)
        try:
            text = self._paligemma_tokenizer.decode(tokens.tolist())
            if 'Action: ' in text:
                pg = self._paligemma_tokenizer.encode(text.split('Action: ')[1].split('|')[0].strip())
                ids = self._act_tokens_to_paligemma_tokens(np.asarray(pg)).tolist()
                count = len(self._fast_tokenizer.bpe_tokenizer.decode(ids))
                last_decode['coefficient_count'] = count
                validate_fast_tokens(self._fast_tokenizer.bpe_tokenizer, ids, action_horizon * action_dim)
                last_decode['valid'] = True
        except Exception as exc:
            last_decode['inspection_error'] = str(exc)
        # Malformed bytes must not become Unicode replacement-codepoint DCT coefficients.
        # Keep the same normalized-zero fallback as existing malformed-shape decoding.
        if not last_decode['valid']:
            return np.zeros((action_horizon, action_dim), dtype=np.float32)
        result = super().extract_actions(tokens, action_horizon, action_dim)
        if not np.isfinite(result).all():
            last_decode.update(valid=False, inspection_error='Nonfinite decoded action')
            return np.zeros((action_horizon, action_dim), dtype=np.float32)
        return result

class AuditedPolicy:
    def __init__(self, policy):
        self.policy = policy
    def infer(self, observation):
        output = self.policy.infer(observation)
        output['action_decode'] = dict(last_decode)
        return output

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--port', type=int, default=8011)
    parser.add_argument('--config', default='pi0_fast_libero_5090_lora')
    parser.add_argument('--residual-controller')
    parser.add_argument('--residual-beta',type=float,default=1.0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, force=True)
    assert jax.default_backend() == 'gpu'
    cfg = get_config(args.config)
    cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, fast_model_tokenizer=AuditedTokenizer))
    metadata={'checkpoint':args.checkpoint,'config':cfg.name,'repo_id':cfg.data.repo_id,
              'decode_validation':DECODE_VALIDATION_VERSION}
    if args.residual_controller:
        from residual_policy import CONFIG
        from residual_policy_factory import create_residual_policy
        if args.config == 'pi0_fast_libero_spatial_lora':
            from libero_residual_policy import create_libero_residual_policy
            policy=create_libero_residual_policy(args.checkpoint,args.residual_controller,args.residual_beta)
        elif args.config == CONFIG:
            policy=create_residual_policy(args.checkpoint,args.residual_controller,args.residual_beta)
        else:
            raise ValueError('Unsupported residual policy config')
        metadata['residual_controller']=policy.metadata
    else:
        policy=AuditedPolicy(create_trained_policy(cfg,args.checkpoint))
    print('MODEL_READY', args.checkpoint, flush=True)
    WebsocketPolicyServer(policy, host='127.0.0.1', port=args.port,
                          metadata=metadata).serve_forever()

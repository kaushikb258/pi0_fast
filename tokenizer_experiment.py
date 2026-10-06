"""Load FAST+ or a future custom tokenizer and run a synthetic encode/decode check. Never trains."""
import argparse
import json
import pathlib
import numpy as np
from transformers import AutoProcessor

parser = argparse.ArgumentParser()
parser.add_argument('--tokenizer', choices=['fast_plus', 'custom'], default='fast_plus')
parser.add_argument('--path', help='Local saved tokenizer directory; required for custom')
args = parser.parse_args()
if args.tokenizer == 'custom' and not args.path:
    parser.error('--path is required for custom; this script does not train or create a tokenizer')
path = pathlib.Path(args.path or '/home/kb/pi0_fast/fast-tokenizer').resolve()
processor = AutoProcessor.from_pretrained(str(path), trust_remote_code=True, local_files_only=True)
actions = np.random.default_rng(0).uniform(-1, 1, size=(1,10,7)).astype(np.float32)
tokens = processor(actions)
# Validate dimensions independently: the upstream decoder can silently fall back to zeros.
assert len(processor.bpe_tokenizer.decode(tokens[0])) == 70
decoded = processor.decode(tokens, time_horizon=10, action_dim=7)
assert decoded.shape == actions.shape and np.isfinite(decoded).all()
rmse = float(np.sqrt(np.mean((actions-decoded)**2)))
assert rmse < 0.1, rmse
print(json.dumps({'tokenizer': args.tokenizer, 'path': str(path), 'shape': list(decoded.shape), 'tokens': len(tokens[0]), 'rmse': rmse, 'status': 'TOKENIZER_ROUNDTRIP_PASS'}, indent=2))

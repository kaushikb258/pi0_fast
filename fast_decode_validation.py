"""Validate FAST byte-level BPE output before interpreting characters as DCT coefficients."""
import json
import numpy as np

DECODE_VALIDATION_VERSION = 'fast-bytelevel-strict-utf8-v1'

# ByteLevel's reversible byte -> Unicode alphabet (GPT-2 convention).
_visible = list(range(ord('!'), ord('~') + 1)) + list(range(161, 173)) + list(range(174, 256))
_BYTE_DECODER = {chr(value): value for value in _visible}
for offset, value in enumerate(value for value in range(256) if value not in _visible):
    _BYTE_DECODER[chr(256 + offset)] = value


def validate_fast_tokens(tokenizer, ids, expected_coefficients):
    """Raise on corrupt encoding; do not impose an arbitrary action magnitude limit."""
    backend = tokenizer.backend_tokenizer
    state = backend.decoder.__getstate__()
    if isinstance(state, bytes): state = state.decode()
    if json.loads(state)['type'] != 'ByteLevel':
        raise ValueError('Unsupported FAST BPE decoder; expected ByteLevel')
    vocab = tokenizer.get_vocab()
    valid_ids = set(vocab.values())
    if any(not isinstance(i, (int, np.integer)) or int(i) not in valid_ids for i in ids):
        raise ValueError('Action token ID outside FAST vocabulary')
    pieces = tokenizer.convert_ids_to_tokens([int(i) for i in ids])
    try:
        raw = bytes(_BYTE_DECODER[c] for piece in pieces for c in piece)
        decoded = raw.decode('utf-8', errors='strict')
    except (KeyError, UnicodeDecodeError) as exc:
        raise ValueError('Malformed UTF-8 in generated FAST byte sequence') from exc
    if '\ufffd' in decoded:
        raise ValueError('Replacement character is not a valid FAST coefficient')
    if len(decoded) != expected_coefficients:
        raise ValueError(f'Expected {expected_coefficients} coefficients, got {len(decoded)}')
    if decoded != tokenizer.decode(ids):
        raise ValueError('Strict and tokenizer byte decoding disagree')
    return decoded

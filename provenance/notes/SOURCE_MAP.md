# Local editable sources

OpenPI checkout: `/home/kb/pi0_fast/openpi`
Official upstream: https://github.com/Physical-Intelligence/openpi
Initial revision: `215abfb217dbac7d5f1273282331b9b1866c0479`

- `src/openpi/models/pi0_fast.py`: pi0-FAST config, loss, autoregressive sampling.
- `src/openpi/models/tokenizer.py`: FASTTokenizer; prompt/state and action token mapping.
- `src/openpi/transforms.py`: TokenizeFASTInputs and ExtractFASTActions.
- `src/openpi/models/gemma_fast.py`: transformer model and decoding cache.
- `src/openpi/training/config.py`: `pi0_fast_libero` and `pi0_fast_libero_low_mem_finetune` (LoRA).
- `src/openpi/policies/libero_policy.py`: LIBERO observation and action mapping.
- `examples/libero/`: data conversion and evaluation examples.

Standalone official FAST tokenizer: `/home/kb/pi0_fast/fast-tokenizer`
Upstream: https://huggingface.co/physical-intelligence/fast
`processing_action_tokenizer.py` contains DCT quantization, BPE encoding and inverse decoding.
The smoke script loads this local directory explicitly. OpenPI's default points to the Hugging Face repo; for experiments pass `fast_tokenizer_path="/home/kb/pi0_fast/fast-tokenizer"` to FASTTokenizer or via the model's `fast_model_tokenizer_kwargs`. Restart Python after code edits. Transformers loads custom processor code through its module cache, so already-running instances do not reload edits.

The model checkpoint and fitted tokenizer vocabulary are coupled. Changing tokenizer behavior later is an experiment and may require corresponding model training. No tokenizer behavior was changed for this installation.

Official base checkpoint: `gs://openpi-assets/checkpoints/pi0_fast_base`.
The two FAST/LIBERO configs use 7 action dimensions, horizon 10, max_token_len 180, and dataset `physical-intelligence/libero`. Both initialize from the base checkpoint. The low-memory config selects `gemma_2b_lora` and disables EMA. No training has been run.

## Tokenizer experiment options

1. **FAST+ (baseline)**: the released universal pretrained tokenizer in `fast-tokenizer`, revision `ec4d7aa71691cac0b8bed6942be45684db2110f4`. This is the artifact named `physical-intelligence/fast` on Hugging Face and already the default used by OpenPI. FAST is the method; FAST+ is the released universal fitted tokenizer, not a separate pi0 model checkpoint.
2. **Custom tokenizer (future)**: edit the local tokenizer source or later fit/save a different tokenizer. No custom tokenizer has been trained. Supply its saved directory explicitly when ready.

After activating pi0:

```bash
python /home/kb/pi0_fast/tokenizer_experiment.py --tokenizer fast_plus
python /home/kb/pi0_fast/tokenizer_experiment.py --tokenizer custom --path /absolute/path/to/saved/tokenizer
```

These commands only perform synthetic encode/decode checks. The custom path must already exist. For a future OpenPI config, use `dataclasses.replace(config.model, fast_model_tokenizer_kwargs={"fast_tokenizer_path": "/absolute/path/to/saved/tokenizer"})` and preserve the remaining LIBERO config settings.

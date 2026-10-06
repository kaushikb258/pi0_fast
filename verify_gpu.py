import json, sys
import importlib.metadata as md
print('Python:', sys.version, flush=True)
for name in ['openpi','jax','jaxlib','flax','torch','torchvision','nvidia-cudnn-cu12','nvidia-cublas-cu12','nvidia-cuda-runtime-cu12','nvidia-cuda-nvcc-cu12']:
    print(name, md.version(name), flush=True)
if sys.argv[1] == 'jax':
    import jax, jax.numpy as jnp
    print('JAX devices:', jax.devices(), flush=True)
    assert jax.default_backend() == 'gpu'
    x = jnp.ones((256,256), dtype=jnp.bfloat16)
    y = jax.jit(lambda a: a @ a)(x).block_until_ready()
    assert y.devices().pop().platform == 'gpu'
    assert bool(jnp.all(y == 256))
    print('JAX GPU BF16 MATMUL PASS:', y.shape, y.devices(), flush=True)
else:
    import torch
    print('Torch CUDA:', torch.version.cuda, 'available:', torch.cuda.is_available(), flush=True)
    print('GPU:', torch.cuda.get_device_name(), 'capability:', torch.cuda.get_device_capability(), 'archs:', torch.cuda.get_arch_list(), flush=True)
    x = torch.ones((256,256),device='cuda',dtype=torch.bfloat16)
    y = x @ x
    torch.cuda.synchronize()
    assert torch.all(y == 256).item()
    print('TORCH GPU BF16 MATMUL PASS:', y.shape, y.device, flush=True)

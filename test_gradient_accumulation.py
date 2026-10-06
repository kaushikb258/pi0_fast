import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from openpi.training.gradient_accumulation import mean_microbatch_gradients
model=nnx.Linear(3,2,rngs=nnx.Rngs(42))
graph,state=nnx.split(model)
x=jnp.arange(24,dtype=jnp.float32).reshape(8,3)/10
y=jnp.sin(x[:,:2])
def lg(rng,batch):
 m=nnx.merge(graph,state)
 return nnx.value_and_grad(lambda m:jnp.mean((m(batch[0])-batch[1])**2))(m)
rng=jax.random.key(4)
base=lg(rng,(x,y))
for steps in (1,4,8):
 out=jax.jit(lambda:mean_microbatch_gradients(lg,rng,(x,y),steps))()
 for a,b in zip(jax.tree.leaves(base),jax.tree.leaves(out)):
  np.testing.assert_allclose(a,b,rtol=2e-6,atol=2e-6)
 tx=optax.chain(optax.clip_by_global_norm(.1),optax.adamw(.001))
 opts=tx.init(state)
 expected=tx.update(base[1],opts,state)
 actual=tx.update(out[1],opts,state)
 for a,b in zip(jax.tree.leaves(expected),jax.tree.leaves(actual)):
  np.testing.assert_allclose(a,b,rtol=2e-6,atol=2e-6)
 print('PASS: accumulation',steps,'matches full-batch loss, gradients, clipped AdamW update and optimizer state')

# Exercise the actual training step, including frozen parameters and optimizer clock.
import dataclasses
import functools
import sys
sys.path.insert(0, '/home/kb/pi0_fast/openpi')
from scripts.train import train_step
from openpi.models.model import BaseModel, Observation
from openpi.training.config import get_config
from openpi.training.utils import TrainState
from openpi.shared.nnx_utils import PathRegex
class TinyModel(BaseModel):
 def __init__(self):
  super().__init__(action_dim=2,action_horizon=1,max_token_len=1)
  self.kernel=nnx.Param(jnp.ones((3,2)))
  self.frozen=nnx.Param(jnp.ones((2,)))
 def compute_loss(self,rng,observation,actions,*,train=False):
  return jnp.mean((observation.state@self.kernel+self.frozen-actions[:,0,:])**2,axis=-1)[:,None]
 def sample_actions(self,rng,observation,**kwargs):
  return (observation.state@self.kernel+self.frozen)[:,None,:]
cfg=dataclasses.replace(get_config('pi0_fast_libero_spatial_lora'),batch_size=8,freeze_filter=PathRegex('frozen'))
graph,params=nnx.split(TinyModel()); tx=optax.chain(optax.clip_by_global_norm(.1),optax.adamw(.001))
state=TrainState(step=jnp.array(0),params=params,model_def=graph,opt_state=tx.init(params.filter(cfg.trainable_filter)),tx=tx,ema_decay=None)
obs=Observation(images={},image_masks={},state=x)
reference=None
for count in (1,4,8):
 output=jax.jit(functools.partial(train_step,dataclasses.replace(cfg,gradient_accumulation_steps=count)))(rng,state,(obs,y[:,None,:]))
 assert int(output[0].step)==1
 np.testing.assert_array_equal(output[0].params['frozen'].value,params['frozen'].value)
 if reference is None: reference=output
 else:
  for a,b in zip(jax.tree.leaves(reference),jax.tree.leaves(output)):
   np.testing.assert_allclose(a,b,rtol=3e-6,atol=3e-6)
 print('PASS actual train_step:',count,'microbatches; same parameters/optimizer/metrics, frozen weights unchanged, step advanced once')

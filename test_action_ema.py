"""CPU checks for action EMA semantics, independent of simulator and policy weights."""
import numpy as np
from action_ema import ActionEMA, action_change_metrics

raw=np.arange(70,dtype=np.float32).reshape(10,7)/10
raw[:,6]=np.where(np.arange(10)%2,1.,-1.)
original=raw.copy()
ema=ActionEMA(.2)
actual=[]
previous=None
# Two chunks ensure the history survives the replanning boundary.
for chunk in (raw[:5],raw[5:]):
 for action in chunk:
  expected=action.copy()
  if previous is not None:
   expected[:6]=.2*previous[:6]+.8*action[:6]
  out=ema.apply(action)
  np.testing.assert_allclose(out,expected,rtol=1e-6)
  assert out[6]==action[6]
  previous=expected
  actual.append(out)
np.testing.assert_array_equal(raw,original)
ema.reset()
np.testing.assert_array_equal(ema.apply(raw[9]),raw[9])
identity=ActionEMA(0)
for a in raw:
 np.testing.assert_array_equal(identity.apply(a),a)
for invalid in (-.1,1,1.1,float('nan'),float('inf')):
 try: ActionEMA(invalid)
 except ValueError: pass
 else: raise AssertionError('Accepted invalid lambda')
assert action_change_metrics([])['transitions']==0
assert action_change_metrics(raw[:1])['translation_delta_rms'] is None
assert action_change_metrics(raw)['transitions']==9
print('PASS: recurrence, cross-chunk history, gripper unchanged, input immutable, episode reset, lambda-0 identity, validation, metrics')

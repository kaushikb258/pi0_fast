"""CPU contract tests; no VLA download, simulator, or experiment training."""
import tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from residual_controller import ResidualMLP,controller_loss,correct_actions
from train_residual_controller import CachedSamples

class ControllerTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(0);torch.set_num_threads(2)
    def test_zero_initialization_and_gripper(self):
        model=ResidualMLP();features=torch.randn(2,4,2048);base=torch.randn(2,10,7)
        correction=model(features)
        self.assertEqual(tuple(correction.shape),(2,10,6));self.assertEqual(correction.abs().max().item(),0.)
        torch.testing.assert_close(correct_actions(base,correction,torch.ones(6)),base,rtol=0,atol=0)
        torch.testing.assert_close(model.layer_weights().sum(),torch.tensor(1.))
        corrected=correct_actions(base,torch.ones_like(correction),torch.arange(1.,7.))
        torch.testing.assert_close(corrected[...,6],base[...,6],rtol=0,atol=0)
    def test_large_features_preserve_range(self):
        from residual_policy import checked_prefix_features
        x=np.zeros((4,2048),np.float32);x[3,0]=100000.;x[2,1]=-100000.
        y=checked_prefix_features(x)
        self.assertEqual(y.dtype,np.float32)
        np.testing.assert_array_equal(x,y)
        self.assertTrue(torch.isfinite(ResidualMLP()(torch.from_numpy(y)[None])).all())
        x[0,0]=np.inf
        with self.assertRaises(FloatingPointError):checked_prefix_features(x)

    def test_padding_mask(self):
        pred=torch.zeros(1,10,6);base=torch.zeros(1,10,7);target=torch.zeros_like(base)
        target[:,2:,:6]=999;mask=torch.zeros(1,10,dtype=torch.bool);mask[:,:2]=True
        loss,mse=controller_loss(pred,base,target,torch.ones(6),mask)
        self.assertEqual(loss.item(),0.);self.assertEqual(mse.item(),0.)
    def test_synthetic_fit_and_layer_gradients(self):
        model=ResidualMLP(feature_dim=16,hidden_dims=(16,8));x=torch.randn(16,4,16)
        target=torch.ones(16,10,6)*.2;optimizer=torch.optim.Adam(model.parameters(),lr=.01)
        initial=((model(x)-target)**2).mean().item()
        for _ in range(30):
            loss=((model(x)-target)**2).mean();optimizer.zero_grad();loss.backward();optimizer.step()
        self.assertLess(((model(x)-target)**2).mean().item(),initial*.2)
        self.assertGreater(model.layer_logits.grad.abs().sum().item(),0.)
        self.assertLessEqual(model(x).abs().max().item(),1.)
    def test_split_and_invalid_filter(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)
            values={'features':np.zeros((4,4,2048),np.float16),'base_actions':np.zeros((4,10,7),np.float32),
                'target_actions':np.zeros((4,10,7),np.float32),'time_mask':np.ones((4,10),bool),
                'valid_decode':np.array([True,False,True,False]),'is_validation':np.array([False,False,True,True]),
                'episode_index':np.array([0,0,1,1])}
            for k,v in values.items():np.save(p/(k+'.npy'),v)
            train=CachedSamples(p,False);val=CachedSamples(p,True)
            self.assertEqual(train.indices.tolist(),[0]);self.assertEqual(val.indices.tolist(),[2])
if __name__=='__main__':unittest.main()

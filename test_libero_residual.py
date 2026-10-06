"""LIBERO-specific contracts; CPU only, no model inference or training."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from libero_residual_data import DATASET, load_metadata, validation_split, read_episode, observation, target_chunk
from libero_residual_policy import CONFIG, LAYERS, DEFAULT_CHECKPOINT, create_libero_residual_policy
from residual_controller import ResidualMLP
from residual_policy import ResidualPolicy

class LiberoResidualTests(unittest.TestCase):
    def test_stratified_split_and_chunks(self):
        episodes=load_metadata();validation=validation_split(episodes)
        self.assertEqual(len(validation),43)
        for task in range(10):
            ids={e['episode_index'] for e in episodes if e['task_index']==task}
            self.assertTrue(ids & validation);self.assertTrue(ids-validation)
        d=read_episode(DATASET,episodes[0]);o=observation(d,0)
        self.assertNotIn('actions',o)
        self.assertEqual(o['observation/image'].shape,(224,224,3))
        target,mask=target_chunk(d['actions'],len(d['actions'])-1)
        self.assertEqual(mask.sum(),1)
        np.testing.assert_array_equal(target,np.repeat(d['actions'][-1:],10,axis=0))

    def test_legacy_transform_roundtrip(self):
        from openpi.training.config import get_config
        from openpi.training.checkpoints import load_norm_stats
        from openpi.transforms import Normalize, Unnormalize, compose
        cfg=get_config(CONFIG);dc=cfg.data.create(cfg.assets_dirs,cfg.model)
        stats=load_norm_stats(DEFAULT_CHECKPOINT/'assets',dc.asset_id)
        for meta in load_metadata()[::43]:
            d=read_episode(DATASET,meta);obs=observation(d,0)
            target,_=target_chunk(d['actions'],0)
            x=compose(dc.data_transforms.inputs)({**obs,'actions':target.copy()})
            x=Normalize(stats,use_quantiles=True)(x)
            x=Unnormalize(stats,use_quantiles=True)(x)
            result=compose(dc.data_transforms.outputs)(x)
            np.testing.assert_allclose(result['actions'],target,atol=3e-6,rtol=3e-6)

    def test_residual_routing_units_gripper_and_fallback(self):
        model=ResidualMLP()
        with torch.no_grad():model.mlp[-1].bias.fill_(.2)
        scales=np.array([.1,.2,.3,.4,.5,.6],np.float32)
        base=np.arange(70,dtype=np.float32).reshape(10,7)/100
        class FakeFrozen:
            checkpoint=str(DEFAULT_CHECKPOINT.resolve())
            feature_code_hash='test'
            valid=True
            def infer_with_features(self,obs):
                return {'actions':base.copy(),'action_decode':{'valid':self.valid}},np.zeros((4,2048),np.float32)
        frozen=FakeFrozen()
        bundle={'config':CONFIG,'layers':LAYERS,'base_checkpoint':frozen.checkpoint,
            'cache_manifest_hash':'test','feature_code_hash':'test','architecture':{'num_layers':4},
            'state_dict':model.state_dict(),'motion_scales':scales.tolist()}
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'test.pt';torch.save(bundle,path)
            with patch('libero_residual_policy.FrozenLiberoFeaturePolicy',return_value=frozen):
                policy=create_libero_residual_policy(DEFAULT_CHECKPOINT,path,beta=1.)
            corrected=policy.infer({})['actions']
            np.testing.assert_allclose(corrected[:,:6],base[:,:6]+np.tanh(.2)*scales,atol=1e-7)
            np.testing.assert_array_equal(corrected[:,6],base[:,6])
            frozen.valid=False
            np.testing.assert_array_equal(policy.infer({})['actions'],base)
            frozen.valid=True;policy.beta=0
            np.testing.assert_array_equal(policy.infer({})['actions'],base)
            bundle['config']='pi0_fast_mimicgen_square_lora';torch.save(bundle,path)
            with self.assertRaises(ValueError):create_libero_residual_policy(DEFAULT_CHECKPOINT,path)

if __name__=='__main__':unittest.main()

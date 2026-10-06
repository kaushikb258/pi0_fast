"""Load the selected four-layer residual controller."""
import torch
from residual_policy import FrozenFeaturePolicy, ResidualPolicy, LAYERS

def controller_layers(bundle):
    layers=bundle.get('layers',LAYERS)
    if layers != LAYERS or bundle['architecture']['num_layers'] != len(LAYERS):
        raise ValueError('This release supports Gemma layers 4, 8, 12, 18 only')
    return list(layers)

def create_residual_policy(checkpoint,controller_path,beta=1.):
    bundle=torch.load(controller_path,map_location='cpu',weights_only=True)
    layers=controller_layers(bundle)
    policy=ResidualPolicy(FrozenFeaturePolicy(checkpoint),controller_path,beta)
    policy.metadata['layers']=layers
    return policy

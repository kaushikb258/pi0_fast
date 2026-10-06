"""Small motion-residual head. The frozen VLA is never part of this module."""
import torch
from torch import nn

class ResidualMLP(nn.Module):
    def __init__(self, feature_dim=2048, num_layers=4, horizon=10,
                 hidden_dims=(256,128), max_normalized_correction=1.0):
        super().__init__()
        self.horizon=horizon
        self.max_normalized_correction=float(max_normalized_correction)
        if self.max_normalized_correction<=0:raise ValueError('Correction bound must be positive')
        self.layer_logits=nn.Parameter(torch.zeros(num_layers))
        self.normalize=nn.LayerNorm(feature_dim,elementwise_affine=False)
        self.mlp=nn.Sequential(nn.Linear(feature_dim,hidden_dims[0]),nn.GELU(),
            nn.Linear(hidden_dims[0],hidden_dims[1]),nn.GELU(),nn.Linear(hidden_dims[1],horizon*6))
        # Starts as exactly the original policy, even with nonzero hidden features.
        nn.init.zeros_(self.mlp[-1].weight);nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, features):
        normalized=self.normalize(features.float())
        weights=self.layer_logits.softmax(dim=0)
        fused=(normalized*weights[None,:,None]).sum(dim=1)
        return torch.tanh(self.mlp(fused)).reshape(-1,self.horizon,6)*self.max_normalized_correction

    def layer_weights(self):return self.layer_logits.softmax(dim=0)


def controller_loss(prediction, base_actions, target_actions, scales, mask, residual_penalty=1e-3):
    """Masked motion MSE in training-data standard-deviation units, plus small residual penalty."""
    target_residual=(target_actions[...,:6]-base_actions[...,:6])/scales
    weights=mask[...,None].to(prediction.dtype)
    denominator=(weights.sum()*6).clamp_min(1)
    mse=((prediction-target_residual).square()*weights).sum()/denominator
    regularizer=(prediction.square()*weights).sum()/denominator
    return mse+residual_penalty*regularizer,mse


def correct_actions(base_actions, normalized_residual, scales, beta=1.0):
    """Only six motion columns change. Units are final simulator commands before EMA."""
    result=base_actions.clone()
    result[...,:6]=result[...,:6]+float(beta)*normalized_residual*scales
    return result

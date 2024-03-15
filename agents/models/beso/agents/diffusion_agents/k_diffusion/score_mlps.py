
from multiprocessing.sharedctypes import Value
import einops

import torch
import hydra
from torch import DictType, nn
from .utils import append_dims
from omegaconf import DictConfig

from agents.models.beso.networks.mlps.mlps import *
from agents.models.beso.networks.mlps.film_mlps import *
from agents.models.beso.networks.mlps.res_layers import *


class GCTimeScoreNetwork(nn.Module):

    def __init__(
            self,
            obs_dim: int,
            action_dim: int,
            hidden_dim: int,
            goal_dim: int,
            embed_fn: DictConfig,
            cond_mask_prob: float,
            num_hidden_layers: int = 1,
            output_dim=1,
            dropout: int = 0,
            activation: str = "ReLU",
            model_style: str = True,
            use_norm: bool = False,
            norm_style: str = 'BatchNorm',
            use_spectral_norm: bool = False,
            device: str = 'cuda',
            goal_conditional: bool = True
    ):
        super(GCTimeScoreNetwork, self).__init__()
        self.network_type = "mlp"
        #  Gaussian random feature embedding layer for time
        self.embed = hydra.utils.call(embed_fn)
        self.time_embed_dim = embed_fn.time_embed_dim
        self.cond_mask_prob = cond_mask_prob
        self.goal_conditional = goal_conditional
        if self.goal_conditional:
            input_dim = self.time_embed_dim + obs_dim + action_dim  + goal_dim
        else:
            input_dim = self.time_embed_dim + obs_dim + action_dim  
        # set up the network
        if model_style:
            self.layers = ResidualMLPNetwork(
                input_dim,
                hidden_dim,
                num_hidden_layers,
                output_dim,
                dropout,
                activation,
                use_spectral_norm,
                use_norm,
                norm_style,
                device
            ).to(device)
        else:
            self.layers = MLPNetwork(
                input_dim,
                hidden_dim,
                num_hidden_layers,
                output_dim,
                dropout,
                activation,
                use_spectral_norm,
                device
            ).to(device)

        # build the activation layer
        self.act = return_activiation_fcn(activation)
        self.device = device
        self.sigma = None
        self.training = True

    def forward(self, state, action, goal, sigma, return_variance=False, uncond=False):
        # Obtain the Gaussian random feature embedding for t
        t = sigma.log() / 4
        embed = self.embed(t)
        # embed = self.embed(t)
        if len(state.shape) == 3:
            embed = einops.rearrange(embed, 'b d -> b 1 d')
            # embed = embed.repeat(1, state.shape[1], 1)
        # during training randomly mask out the goal
        # to train the conditional model with classifier-free guidance wen need 
        # to 0 out some of the conditional during training with a desrired probability
        # it is usually in the range of 0,1 to 0.2 
        if self.training and self.goal_conditional:
            goal = self.mask_cond(goal)
        # we want to use unconditional sampling during clasisfier free guidance
        if uncond:
            goal = torch.zeros_like(goal)   # goal
        if self.goal_conditional:
            x = torch.cat([state, action, goal, embed], dim=-1) 
        else:
            x = torch.cat([state, action, embed], dim=-1) 
        x = self.layers(x) 
        if return_variance:
            return x, x.flatten(1).mean(1)
        else:
            return x  # / marginal_prob_std(t, self.sigma, self.device)[:, None]

    def mask_cond(self, cond, force_mask=False):
        bs, t, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones((bs, t, d), device=cond.device) * self.cond_mask_prob)# .view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond

    def get_params(self):
        return self.parameters()


class FiLMTimeScoreNetwork(nn.Module):

    def __init__(
            self,
            obs_dim: int,
            action_dim: int,
            hidden_dim: int,
            goal_dim: int,
            embed_fn: DictConfig,
            cond_mask_prob: float,
            num_hidden_layers: int = 1,
            output_dim=1,
            dropout: int = 0,
            activation: str = "ReLU",
            model_style: str = 'residual',
            use_norm: bool = False,
            # embed_mode: bool=False,
            norm_style: str = 'BatchNorm',
            use_spectral_norm: bool = False,
            device: str = 'cuda',
            goal_conditional: bool = True
    ):
        super(FiLMTimeScoreNetwork, self).__init__()
        self.network_type = "mlp"
        #  Gaussian random feature embedding layer for time
        self.embed = hydra.utils.call(embed_fn)
        self.time_embed_dim = embed_fn.time_embed_dim
        self.cond_mask_prob = cond_mask_prob
        self.goal_conditional = goal_conditional
        input_dim = obs_dim + action_dim  
        # set up the network
        condition_dim = goal_dim
        if model_style:
            self.layers = FiLMResidualMLPNetworkV2(
                input_dim,
                condition_dim,
                hidden_dim,
                num_hidden_layers,
                output_dim,
                dropout,
                activation,
                use_spectral_norm,
                use_norm,
                norm_style,
                device
            ).to(device)
        else:
            self.layers = FiLMMLPNetwork(
                input_dim,
                condition_dim,
                hidden_dim,
                num_hidden_layers,
                output_dim,
                dropout,
                activation,
                use_spectral_norm,
                device
            ).to(device)

        # build the activation layer
        self.act = return_activiation_fcn(activation)
        self.device = device
        self.sigma = None
        self.training = True

    def forward(self, state, action, goal, sigma, return_variance=False, uncond=False):
        # Obtain the Gaussian random feature embedding for t
        t = sigma.log() / 4
        embed = self.embed(t)
        # embed = self.embed(t)
        if len(state.shape) == 3:
            embed = einops.rearrange(embed, 'b d -> b 1 d')
        ## FIXME: Expand embed to match the state shape
        elif len(state.shape) == 4 and len(embed.shape) < 4:
            embed = einops.rearrange(embed, 'b d -> b 1 1 d')
        # during training randomly mask out the goal
        # to train the conditional model with classifier-free guidance wen need 
        # to 0 out some of the conditional during training with a desrired probability
        # it is usually in the range of 0,1 to 0.2 
        if self.training:
            goal = self.mask_cond(goal)
        # we want to use unconditional sampling during clasisfier free guidance
        if uncond:
            goal = torch.zeros_like(goal)   # goal

        cond_embed = goal + embed
        x = torch.cat([state, action], dim=-1)

        x = self.layers(x, cond_embed)


        if return_variance:
            return x, x.flatten(1).mean(1)
        else:
            return x  # / marginal_prob_std(t, self.sigma, self.device)[:, None]

    def mask_cond(self, cond, force_mask=False):
        bs, t, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones((bs, t, d), device=cond.device) * self.cond_mask_prob)# .view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond

    def get_params(self):
        return self.parameters()
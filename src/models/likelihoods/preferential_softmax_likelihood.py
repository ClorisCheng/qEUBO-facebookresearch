#!/usr/bin/env python3

import torch
from botorch.sampling import SobolQMCNormalSampler
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.distributions import base_distributions
from gpytorch.likelihoods import Likelihood
from torch.distributions.distribution import Distribution

class PreferentialSoftmaxLikelihood(Likelihood):
    r"""
    Implements the softmax likelihood used for GP-based preference learning.

    .. math::
        p(\mathbf y \mid \mathbf f) = \text{Softmax} \left( \mathbf f \right)

    :param int num_alternatives: Number of alternatives (i.e., q).
    """

    def __init__(self, num_alternatives):
        super().__init__()
        self.num_alternatives = num_alternatives
        self.noise = torch.tensor(1e-4)  # This is only used to draw RFFs-based
        # samples. We set it close to zero because we want noise-free samples
        self.sampler = SobolQMCNormalSampler(sample_shape=512)  # This allows for
        # SAA-based optimization of the ELBO

    def _draw_likelihood_samples(
        self, function_dist, *args, sample_shape=None, **kwargs
    ):
        function_samples = self.sampler(GPyTorchPosterior(function_dist)).squeeze(-1)
        return self.forward(function_samples, *args, **kwargs)

    def forward(self, function_samples, *params, **kwargs):
        function_samples = function_samples.reshape(
            function_samples.shape[:-1]
            + torch.Size(
                (
                    int(function_samples.shape[-1] / self.num_alternatives),
                    self.num_alternatives,
                )
            )
        )  # Reshape samples as if they came from a multi-output model (with `q` outputs)
        num_alternatives = function_samples.shape[-1]

        if num_alternatives != self.num_alternatives:
            raise RuntimeError("There should be %d points" % self.num_alternatives)

        # res = base_distributions.Categorical(logits=function_samples)  # Passing the
        # # function values as logits recovers the softmax likelihood
        # return res

        res = MonteCarloLikelihood(function_samples)
        return res

class MonteCarloLikelihood(Distribution):
    def __init__(self, f_vals, mc_samples=1000, mu=0, beta=1, temperature=0.01):
        super().__init__()
        assert(f_vals.shape[-1]) > 2 # at least 3 attributes
        self.f_vals = f_vals
        self.logits = f_vals - f_vals.logsumexp(dim=-1, keepdim=True)
        self.mc_samples = mc_samples
        self.mu = mu
        self.beta = beta
        self.temperature = temperature
        
    def log_prob(self, value, reuse_across_points=False):
        '''
        Args:
            value: (n, ) tensor of indices
            reuse_across_points: if True, reuse the same epsilon across all points
        Returns:
            log_prob: (sampler_size, n) tensor of log probabilities
        '''
        q = self.f_vals.shape[-1]
        n = self.f_vals.shape[-2]
        sampler_size = self.f_vals.shape[-3]
        
        if reuse_across_points:
            eps_smpl = torch.rand(self.mc_samples, q)
            eps_smpl = eps_smpl[None, ...].repeat((n, 1, 1)) # (n, mc_samples, q)
        else:
            eps_smpl = torch.rand(n, self.mc_samples, q)
                
        # Gumbel: mu - beta * log(-log(U))
        gumbel_param = 0
        eps_smpl = -torch.log(-torch.log(eps_smpl)) + gumbel_param # (n, mc_samples, q)

        # add extra dimension to f_vals to broadcast across mc_samples
        f_vals = self.f_vals.reshape((sampler_size, n, 1, q)).repeat((1, 1, self.mc_samples, 1)) # (sampler_size, n, mc_samples, q)
        # reuse epsilon across sampler_size
        eps_smpl = eps_smpl.reshape((1, n, self.mc_samples, q)).repeat((sampler_size, 1, 1, 1)) # (sampler_size, n, mc_samples, q)
        y_vals = f_vals + eps_smpl
        like = torch.softmax(y_vals / self.temperature, dim=-1) # (sampler_size, n, q)
        like = torch.mean(like, dim=-2) # (sampler_size, n, q)
        # expand index to select repeatedly across sampler_size
        value = value[None, :, None].expand(sampler_size, -1, -1) # (sampler_size, n, 1)
        like = torch.gather(like, 2, value).squeeze(-1) # (sampler_size, n)
        
        return torch.log(like)
    

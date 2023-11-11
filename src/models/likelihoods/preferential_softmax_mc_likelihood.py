#!/usr/bin/env python3

import torch
from botorch.sampling import SobolQMCNormalSampler
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.distributions import base_distributions
from gpytorch.likelihoods import Likelihood
from torch.distributions.distribution import Distribution

class PreferentialSoftmaxMCLikelihood(Likelihood):
    r"""
    Implements the softmax likelihood used for GP-based preference learning.

    .. math::
        p(\mathbf y \mid \mathbf f) = \text{Softmax} \left( \mathbf f \right)

    :param int num_alternatives: Number of alternatives (i.e., q).
    """

    def __init__(self, num_alternatives, **kwargs):
        super().__init__()
        self.num_alternatives = num_alternatives
        self.noise = torch.tensor(1e-4)  # This is only used to draw RFFs-based
        # samples. We set it close to zero because we want noise-free samples
        self.sampler = SobolQMCNormalSampler(sample_shape=512)  # This allows for
        # SAA-based optimization of the ELBO
        self.eps = None
        self.base_distribution_params = kwargs.get("base_distribution_params", {})



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
        
        mc_samples = self.base_distribution_params.get("mc_samples", 1000)
        mu = self.base_distribution_params.get("mu", 0)
        beta = self.base_distribution_params.get("beta", 1)
        gumbel_param = self.base_distribution_params.get("gumbel_param", 0)
        temperature = self.base_distribution_params.get("temperature", 0.01)
        if self.eps is None:
            self.eps = torch.rand(mc_samples, num_alternatives)
        else:
            assert num_alternatives == self.eps.shape[-1]
        

        
        return MonteCarloLikelihood(function_samples, self.eps, mc_samples=mc_samples, mu=mu, beta=beta, gumbel_param=gumbel_param, temperature=temperature)



class MonteCarloLikelihood(Distribution):
    def __init__(self, f_vals, eps, mc_samples=1000, mu=0, beta=1, gumbel_param = 0, temperature=0.01):
        super().__init__(validate_args=False)
        assert(f_vals.shape[-1]) > 2 # at least 3 attributes

        self.f_vals = f_vals
        self.logits = f_vals - f_vals.logsumexp(dim=-1, keepdim=True)
        self.q = f_vals.shape[-1]
        self.n = f_vals.shape[-2]
        self.sampler_size = f_vals.shape[-3]
        self.mc_samples = mc_samples
        self.mu = mu
        self.beta = beta
        self.temperature = temperature
        self.gumbel_param = gumbel_param
        # self.arg_constraints = {}
        self.eps = - self.beta * torch.log(-torch.log(eps)) + self.gumbel_param # (mc_samples, q)


        
        
    def log_prob(self, value, reuse_across_points=True):
        '''
        Args:
            value: (n, ) tensor of indices
            reuse_across_points: if True, reuse the same epsilon across all points
        Returns:
            log_prob: (sampler_size, n) tensor of log probabilities
        '''
        
        assert value.shape[0] == self.n

        # if reuse_across_points:
        #     eps_smpl = torch.rand(self.mc_samples, q)
        #     eps_smpl = eps_smpl[None, ...].repeat((n, 1, 1)) # (n, mc_samples, q)
        # else:
        #     eps_smpl = torch.rand(n, self.mc_samples, q)
                
        # Gumbel: mu - beta * log(-log(U))
        eps_smpl = self.eps[None, ...].repeat((self.n, 1, 1)) # (n, mc_samples, q)
        
        # add extra dimension to f_vals to broadcast across mc_samples
        f_vals = self.f_vals.reshape((self.sampler_size, self.n, 1,self.q)).repeat((1, 1, self.mc_samples, 1)) # (sampler_size, n, mc_samples, q)
        # reuse epsilon across sampler_size
        eps_smpl = eps_smpl.reshape((1, self.n, self.mc_samples, self.q)).repeat((self.sampler_size, 1, 1, 1)) # (sampler_size, n, mc_samples, q)
        y_vals = f_vals + eps_smpl
        like = torch.softmax(y_vals / self.temperature, dim=-1) # (sampler_size, n, q)
        like = torch.mean(like, dim=-2) # (sampler_size, n, q)
        # expand index to select repeatedly across sampler_size
        value = value[None, :, None].expand(self.sampler_size, -1, -1) # (sampler_size, n, 1)
        like = torch.gather(like, 2, value).squeeze(-1) # (sampler_size, n)
        
        return torch.log(like)
    

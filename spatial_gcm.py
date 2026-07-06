"""
Spatial GCM.

This code implements spatial causal inference, based on the GCM in Fig. 1b.
It reproduces the results in Section 5.3 of the paper.

It compares three models:
- outcome_gcm: spatial geometric causal model with confounder
- outcome_gm: spatial geometric model without confounder
- outcome_cm: conventional causal model (linear)

Usage:
    python spatial_gcm --n_experiments 10 --domains "(0.5,0.5);(0.7,0.7);(0.86,0.86);(1.0,1.0)" --output results.pkl

Loading results for plotting (in a notebook):
    import pickle
    with open('results.pkl', 'rb') as f:
        data = pickle.load(f)
    
    # Access results
    for exp in data['experiments']:
        true_effect = exp['true_treatment_effect']
        for domain_result in exp['results_by_domain']:
            domain = domain_result['domain']
            for model_name, model_results in domain_result['models'].items():
                if 'error' not in model_results:
                    posterior_mean = model_results['posterior_mean']
                    abs_error = model_results['abs_error']
                    ci_5 = model_results['ci_5']
                    ci_95 = model_results['ci_95']
                    covers_true = model_results['covers_true']
"""

import numpy as np
import jax.numpy as jnp
from jax import random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_sample, init_to_value, SVI, Trace_ELBO
import os
import time
import argparse
import pickle
from typing import List, Tuple, Dict

from numpyro.infer.autoguide import AutoDelta


# ============================================================================
# Data Generation Process
# ============================================================================
def gaussian_effect(beta_weight, beta_scale, covariate, dists, domain=(1.,1.)):
     """
     Compute the effect of a covariate on the mean of a Gaussian process.
     Uses Monte Carlo integration to approximate the integral over the domain.
     """
     return jnp.mean(beta_weight * jnp.exp(-dists**2 / (2 * beta_scale**2)) 
                    * covariate[None, :], axis=1) * domain[0] * domain[1]

def sample_gaussian(key, dists, scale=1.0, weight=1.0):
    """
    Sample from multivariate Gaussian using x = A @ z, where z is N(0, 1) and 
    A is parameterized to fall off as a Gaussian kernel with distance.
    """
    A = weight * jnp.exp(-dists**2 / (2 * scale**2))
    z = random.normal(key, (A.shape[1],))
    return jnp.matmul(A, z)

def dist_matrix(locs):
    """
    Compute the distance matrix for a set of locations.
    """
    return jnp.linalg.norm(locs[:, None, :] - locs[None, :, :], axis=-1)

# Fixed simulation settings.
INTERFERENCE_SCALE = 0.02 # Interference scale.
NOISE_WEIGHT = 0.05 # Noise weight.
CONFOUNDER_WEIGHT = 100. # Latent confounder-observed confounder weight.
TREATMENT_WEIGHT = -100. # Confounder-treatment weight.
BETA_WEIGHT = 50. # Treatment-outcome weight.
GAMMA_WEIGHT = 100. # Confounder-outcome weight.

def dgp(n_samples, key):
     """
     Generate data from the GCM.
     """
     # Irregular sampling locations
     key, subkey = random.split(key)
     loc = random.uniform(subkey, (n_samples, 2))
     dists = dist_matrix(loc)

     # Latent confounder.
     key, subkey = random.split(key)
     u = sample_gaussian(subkey, dists, INTERFERENCE_SCALE/2, 1.)

     # Observed confounder.
     key, subkey = random.split(key)
     x = (sample_gaussian(subkey, dists, INTERFERENCE_SCALE/2, NOISE_WEIGHT) +
          gaussian_effect(CONFOUNDER_WEIGHT, INTERFERENCE_SCALE, u, dists))

     # Observed treatment.
     key, subkey = random.split(key)
     a = (sample_gaussian(subkey, dists, INTERFERENCE_SCALE/2, NOISE_WEIGHT) +
          gaussian_effect(TREATMENT_WEIGHT, INTERFERENCE_SCALE, u, dists))

     # Observed outcome.
     key, subkey = random.split(key)
     y = (sample_gaussian(subkey, dists, INTERFERENCE_SCALE/2, NOISE_WEIGHT) +
          gaussian_effect(BETA_WEIGHT, INTERFERENCE_SCALE, a, dists) + 
          gaussian_effect(GAMMA_WEIGHT, INTERFERENCE_SCALE, x, dists))
     
     # True treatment effect, computed analytically from the Gaussian integral.
     treatment_effect = BETA_WEIGHT * np.sqrt(2*np.pi) * INTERFERENCE_SCALE
     
     return loc, u, x, a, y, treatment_effect


def subsample_data(loc, u, x, a, y, domain=(1., 1.)):
    """Subsample to data within a domain."""
    ind = (loc[:, 0] < domain[0]) & (loc[:, 1] < domain[1])
    loc = loc[ind]
    u = u[ind]
    x = x[ind]
    a = a[ind]
    y = y[ind]
    return loc, u, x, a, y


def get_cholesky_via_qr_positive(A):
    """
    Compute the Cholesky factor L such that A @ A.T = L @ L.T
    using QR decomposition, while ensuring the diagonal of L is positive.
    """
    # 1. Compute QR of A.T
    Q, R = jnp.linalg.qr(A.T, mode='reduced')
    
    # 2. Get initial L
    L = R.T
    
    # 3. Enforce positive diagonal
    diag_L = jnp.diag(L)
    sign_correction = jnp.where(diag_L < 0, -1., 1.)
    L = L * sign_correction
    
    return L


# ============================================================================
# Models.
# ============================================================================
def outcome_gcm(dists, L, a, x, y=None, domain=(1.,1.)):
    """
    Spatial Geometric Causal Model.
    This model follows the structure in Fig. 1b of the paper.

    It takes as input 
    - the distance matrix dists
    - the Cholesky factor of the noise covariance matrix (precomputed to speed up computation)
    - the observed treatment a
    - the observed confounder x
    - the observed outcome y
    - the domain of the data (subset of the unit square)

    It samples the following parameters:
    - beta_weight: the weight of the treatment effect
    - gamma_weight: the weight of the confounder effect
    - noise_weight: the noise weight

    """
    # Prior on the treatment effect weight.
    beta_weight = numpyro.sample('beta_weight', dist.Normal(0, 100))
    # Prior on the confounder effect weight.
    gamma_weight = numpyro.sample('gamma_weight', dist.Normal(0, 100))
    # Compute the mean.
    mu_y = (gaussian_effect(beta_weight, INTERFERENCE_SCALE, a, dists, domain) + 
            gaussian_effect(gamma_weight, INTERFERENCE_SCALE, x, dists, domain))
    
    # Prior on the noise weight.
    noise_weight = numpyro.sample('noise_weight', dist.Normal(0, 10))

    # Sample the outcome y as a Gaussian process.
    numpyro.sample('y', dist.MultivariateNormal(mu_y, scale_tril=noise_weight * L), obs=y)


def outcome_gm(dists, L, a, y=None, domain=(1.,1.)):
    """
    Geometric model that does not account for confounding.

    It takes as input 
    - the distance matrix dists
    - the Cholesky factor of the noise covariance matrix (precomputed to speed up computation)
    - the observed treatment a
    - the observed outcome y
    - the domain of the data (subset of the unit square)

    It samples the following parameters:
    - beta_weight: the weight of the treatment effect
    - noise_weight: the noise weight
    """
    # Prior on the treatment effect weight and scale.
    beta_weight = numpyro.sample('beta_weight', dist.Normal(0, 100))
    # Compute the mean.
    mu_y = gaussian_effect(beta_weight, INTERFERENCE_SCALE, a, dists, domain)
    
    # Prior on the noise weight.
    noise_weight = numpyro.sample('noise_weight', dist.Normal(0, 10))

    # Sample the outcome y as a Gaussian process.
    numpyro.sample('y', dist.MultivariateNormal(mu_y, scale_tril=noise_weight * L), obs=y)


def outcome_cm(a, x, y=None):
    """
    Conventional causal model that does not account for spatial dependence.

    It takes as input
    - the observed treatment a
    - the observed confounder x
    - the observed outcome y

    It samples the following parameters:
    - beta_weight: the weight of the treatment effect
    - gamma_weight: the weight of the confounder effect
    - noise_weight: the noise weight
    """
    # Prior on the treatment effect weight.
    beta_weight = numpyro.sample('beta_weight', dist.Normal(0, 100))
    # Prior on the confounder effect weight
    gamma_weight = numpyro.sample('gamma_weight', dist.Normal(0, 100))
    # Compute the mean.
    mu_y = beta_weight * a + gamma_weight * x
    
    # Prior on the noise weight.
    noise_weight = numpyro.sample('noise_weight', dist.Normal(0, 10))

    # Sample the outcome y from a normal distribution.
    numpyro.sample('y', dist.Normal(mu_y, noise_weight), obs=y)


def compute_treatment_effect_from_samples(beta_weight_samples, model_name):
    """
    Compute treatment effect from beta_weight samples.
    
    For the geometric models (GCM and GM) we must integrate over the interference effects: 
    treatment_effect = beta_weight * sqrt(2*pi) * INTERFERENCE_SCALE

    For the conventional causal model (CM) we can simply use the treatment effect weight.
    treatment_effect = beta_weight
    """
    if model_name in ['outcome_gcm', 'outcome_gm']:
        return beta_weight_samples * np.sqrt(2*np.pi) * INTERFERENCE_SCALE
    else:  # outcome_cm
        return beta_weight_samples

# ============================================================================
# Run experiments
# ============================================================================
def run_single_experiment(experiment_id, n_samples, domains, key, 
                          num_warmup=100, num_samples=200, 
                          svi_init=False, svi_num_steps=10000,
                          svi_step_size=0.000005,
                          only_gcm=False):
    """
    Generate data and run inference for each model, across increasing data subsets.

    Return:
        Dictionary with results for this experiment
    """
    results = {
        'experiment_id': experiment_id,
        'true_treatment_effect': None,
        'domains': domains,
        'results_by_domain': []
    }
    
    # Step 1: Generate data with fresh random seed
    key, subkey = random.split(key)
    loc, u, x, a, y, true_effect = dgp(n_samples, subkey)
    results['true_treatment_effect'] = float(true_effect)

    # Step 2: For each domain, subset data.
    for domain in domains:
        domain_results = {
            'domain': domain,
            'models': {}
        }
        
        # Subsample data to the domain
        loc_sub, u_sub, x_sub, a_sub, y_sub = subsample_data(loc, u, x, a, y, domain)

        # Precompute distance matrix and Cholesky factor for geometric models
        dists = dist_matrix(loc_sub)
        noise_scale = INTERFERENCE_SCALE/2
        A = jnp.exp(-dists**2 / (2 * noise_scale**2))
        eps = 0.01 # Small regularization term to ensure positive definiteness
        L = get_cholesky_via_qr_positive(jnp.concatenate([A, eps * jnp.eye(A.shape[0])], axis=1))

        # Step 3: Run inference for each model.
        models = ['outcome_gcm', 'outcome_gm', 'outcome_cm'] if not only_gcm else ['outcome_gcm']
        for model_name in models:
            try:
                # Step 3.1: Obtain the MAP estimate to initialize the posterior sampler.
                if svi_init:
                    key, subkey = random.split(key)
                    optimizer = numpyro.optim.Adam(step_size=svi_step_size)
                    guide = AutoDelta(eval(model_name))
                    svi = SVI(eval(model_name), guide, optimizer, loss=Trace_ELBO())
                    if model_name == 'outcome_gcm':
                        svi_result = svi.run(subkey, svi_num_steps, dists, L, a_sub, x_sub, y_sub, domain=domain)
                        init_strategy = init_to_value(values={
                                'beta_weight': svi_result.params['beta_weight_auto_loc'],
                                'gamma_weight': svi_result.params['gamma_weight_auto_loc'],
                                'noise_weight': svi_result.params['noise_weight_auto_loc']})
                    elif model_name == 'outcome_gm':
                        svi_result = svi.run(subkey, svi_num_steps, dists, L, a_sub, y_sub, domain=domain)
                        init_strategy = init_to_value(values={
                                'beta_weight': svi_result.params['beta_weight_auto_loc'],
                                'noise_weight': svi_result.params['noise_weight_auto_loc']})
                    elif model_name == 'outcome_cm':
                        svi_result = svi.run(subkey, svi_num_steps, a_sub, x_sub, y_sub)
                        init_strategy = init_to_value(values={
                                'beta_weight': svi_result.params['beta_weight_auto_loc'],
                                'gamma_weight': svi_result.params['gamma_weight_auto_loc'],
                                'noise_weight': svi_result.params['noise_weight_auto_loc']})
                    else:
                        raise ValueError(f"Unknown model: {model_name}")
                    print('SVI treatment effect: ', model_name, ':', compute_treatment_effect_from_samples(
                        svi_result.params['beta_weight_auto_loc'], model_name))
                else:
                    init_strategy = init_to_sample
                key, subkey = random.split(key)
                kernel = NUTS(eval(model_name), init_strategy=init_strategy)
                mcmc = MCMC(
                    kernel,
                    num_warmup=num_warmup,
                    num_samples=num_samples,
                    num_chains=1)
                key, subkey = random.split(key)
                if model_name == 'outcome_gcm':
                    mcmc.run(subkey, dists, L, a_sub, x_sub, y_sub, domain=domain)
                elif model_name == 'outcome_gm':
                    mcmc.run(subkey, dists, L, a_sub, y_sub, domain=domain)
                elif model_name == 'outcome_cm':
                    mcmc.run(subkey, a_sub, x_sub, y_sub)
                samples = mcmc.get_samples()
                beta_weight_samples = np.array(samples['beta_weight'])
                treatment_effect_samples = compute_treatment_effect_from_samples(
                    beta_weight_samples, model_name)
                # Compute statistics
                posterior_mean = float(np.mean(treatment_effect_samples))
                ci_5 = float(np.percentile(treatment_effect_samples, 5))
                ci_95 = float(np.percentile(treatment_effect_samples, 95))
                abs_error = float(np.abs(posterior_mean - true_effect))
                covers_true = (ci_5 <= true_effect <= ci_95)
                
                domain_results['models'][model_name] = {
                    'posterior_mean': posterior_mean,
                    'abs_error': abs_error,
                    'ci_5': ci_5,
                    'ci_95': ci_95,
                    'covers_true': covers_true,
                    'n_samples': len(treatment_effect_samples)
                }
                print('Model: ', model_name, 'Posterior mean: ', posterior_mean, 'True effect: ', true_effect)
                print('CI 5: ', ci_5, 'CI 95: ', ci_95)
                print('Absolute error: ', abs_error)
                print('Coverage: ', covers_true)
            except Exception as e:
                print(f"Error running {model_name} for domain {domain} in experiment {experiment_id}: {e}")
                domain_results['models'][model_name] = {
                    'error': str(e)
                }
        
        results['results_by_domain'].append(domain_results)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Run spatial GCM experiments')
    parser.add_argument('--n_experiments', type=int, default=10,
                        help='Number of experiments to run (default: 10)')
    parser.add_argument('--n_samples', type=int, default=2000,
                        help='Number of data samples to generate (default: 2000)')
    parser.add_argument('--domains', type=str, default='(0.5,0.5);(0.7,0.7);(0.86,0.86);(1.0,1.0)',
                        help='Semicolon-separated list of domain tuples, default: (0.5,0.5);(0.7,0.7);(0.86,0.86);(1.0,1.0)')
    parser.add_argument('--num_warmup', type=int, default=100,
                        help='Number of warmup samples for MCMC (default: 100)')
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='Number of posterior samples for MCMC (default: 1000)')
    parser.add_argument('--output', type=str, default='spatial_gcm_results.pkl',
                        help='Output file path (default: spatial_gcm_results.pkl)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--svi_init', type=bool, default=False,
                        help='Use SVI to initialize the parameters (default: False)')
    parser.add_argument('--svi_num_steps', type=int, default=10000,
                        help='Number of steps for SVI (default: 10000)')
    parser.add_argument('--svi_step_size', type=float, default=0.000005,
                        help='Step size for SVI (default: 0.000005)')
    parser.add_argument('--only_gcm', action='store_true',
                        help='Only run the GCM (default: False)')
    args = parser.parse_args()
    
    # Parse domains
    domains = []
    for domain_str in args.domains.split(';'):
        domain_str = domain_str.strip().strip('()')
        parts = domain_str.split(',')
        if len(parts) != 2:
            raise ValueError(f"Invalid domain format: {domain_str}. Expected format: (x,y)")
        domains.append((float(parts[0]), float(parts[1])))
    
    print(f"Running {args.n_experiments} experiments")
    print(f"Data samples per experiment: {args.n_samples}")
    print(f"Domains: {domains}")
    print(f"MCMC: {args.num_warmup} warmup, {args.num_samples} samples")
    print(f"Output file: {args.output}")
    print()
    
    # Initialize random key
    key = random.key(args.seed)
    
    # Run experiments
    all_results = []
    start_time = time.time()
    
    for exp_id in range(args.n_experiments):
        print(f"Running experiment {exp_id + 1}/{args.n_experiments}...")
        exp_start = time.time()
        
        key, subkey = random.split(key)
        results = run_single_experiment(
            exp_id, args.n_samples, domains, subkey,
            args.num_warmup, args.num_samples,
            args.svi_init, args.svi_num_steps, args.svi_step_size,
            args.only_gcm
        )
        all_results.append(results)
        
        exp_time = time.time() - exp_start
        print(f"  Completed in {exp_time:.2f} seconds")
    
    total_time = time.time() - start_time
    print(f"\nAll experiments completed in {total_time:.2f} seconds")
    
    # Save results
    output_data = {
        'experiments': all_results,
        'config': {
            'n_experiments': args.n_experiments,
            'n_samples': args.n_samples,
            'domains': domains,
            'num_warmup': args.num_warmup,
            'num_samples': args.num_samples,
            'seed': args.seed
        }
    }
    
    with open(args.output, 'wb') as f:
        pickle.dump(output_data, f)
    
    print(f"Results saved to {args.output}")


if __name__ == '__main__':
    main()

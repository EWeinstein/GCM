"""
Array GCM.

This code implements array causal inference, based on the GCM in Fig. 1f.
It reproduces the results in Section 5.4 of the paper.

It compares three models:
- outcome_gcm: array geometric causal model
- outcome_gm: array geometric model (misspecified causal graph)
- outcome_cm: conventional causal model (no latent variables)

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
                    ci_25 = model_results['ci_25']
                    ci_975 = model_results['ci_975']
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
import matplotlib.pyplot as plt
from numpyro.infer.autoguide import AutoDelta, AutoNormal

# ============================================================================
# Data Generation Process
# ============================================================================


NOISE_WEIGHT = 0.1
OFFSET_NOISE_WEIGHT = 2
CONFOUNDER_TREATMENT_WEIGHT = -1. # Confounder-treatment weight.
CONFOUNDER_OUTCOME_WEIGHT = 1. # Confounder-outcome weight.
TREATMENT_MEDIATOR_WEIGHT = 1. # Treatment-mediator weight.
MEDIATOR_OUTCOME_WEIGHT = 1. # Mediator-outcome weight.

def dgp(n_samples, key):
    """
    Generate data from the GCM in Fig. 1c,
    with linear-Gaussian structural equations.
    """
    # Latent confounder.
    key, subkey = random.split(key)
    u_alpha = random.normal(subkey, (n_samples, 1))
    u_beta = random.normal(subkey, (1, n_samples))
    u = u_alpha + u_beta

    # Treatment.
    key, subkey = random.split(key)
    a_alpha = random.normal(subkey, (n_samples, 1))
    a_beta = random.normal(subkey, (1, n_samples))
    a = (u * CONFOUNDER_TREATMENT_WEIGHT 
         + (a_alpha + a_beta) * OFFSET_NOISE_WEIGHT
         + random.normal(subkey, (n_samples, n_samples)) * NOISE_WEIGHT)

    # Mediator.
    key, subkey = random.split(key)
    x_alpha = random.normal(subkey, (n_samples, 1))
    x_beta = random.normal(subkey, (1, n_samples))
    x = (a * TREATMENT_MEDIATOR_WEIGHT 
         + (x_alpha + x_beta) * OFFSET_NOISE_WEIGHT
         + random.normal(subkey, (n_samples, n_samples)) * NOISE_WEIGHT)

    # Outcome.
    key, subkey = random.split(key)
    y_alpha = random.normal(subkey, (n_samples, 1))
    y_beta = random.normal(subkey, (1, n_samples))
    y = (x * MEDIATOR_OUTCOME_WEIGHT 
         + u * CONFOUNDER_OUTCOME_WEIGHT
         + (jnp.abs(y_alpha) + jnp.abs(y_beta)) * OFFSET_NOISE_WEIGHT
         + random.normal(subkey, (n_samples, n_samples)) * NOISE_WEIGHT)

    # True treatment effect, computed analytically.
    true_effect = TREATMENT_MEDIATOR_WEIGHT * MEDIATOR_OUTCOME_WEIGHT
    return u, a, x, y, true_effect

def subsample_data(u, a, x, y, domain=(10, 10)):
    """Subsample to data within the domain."""
    u = u[:domain[0]][:, :domain[1]]
    a = a[:domain[0]][:, :domain[1]]
    x = x[:domain[0]][:, :domain[1]]
    y = y[:domain[0]][:, :domain[1]]
    return u, a, x, y


# ============================================================================
# Models.
# ============================================================================

def outcome_gcm(a, x=None, y=None):
    """
    Array Geometric Causal Model.
    """
    # Prior on the treatment-mediator weight.
    tm_weight = numpyro.sample('tm_weight', dist.Normal(0, 100))

    # Prior on mediator latents.
    x_alpha = numpyro.sample('x_alpha', dist.Normal(jnp.zeros(a.shape[0]), 100))
    x_beta = numpyro.sample('x_beta', dist.Normal(jnp.zeros(a.shape[1]), 100))

    # Sample the mediator.
    numpyro.sample('x', dist.Normal(a * tm_weight + x_alpha[:, None] + x_beta[None, :], 
                                    NOISE_WEIGHT), obs=x)
    

    # Prior on the mediator-outcome weight.
    mo_weight = numpyro.sample('mo_weight', dist.Normal(0, 100))
    # Prior on the treatment-outcome weight.
    to_weight = numpyro.sample('to_weight', dist.Normal(0, 100))
    # Prior on the outcome latents.
    y_alpha = numpyro.sample('y_alpha', dist.Normal(jnp.zeros(a.shape[0]), 100))
    y_beta = numpyro.sample('y_beta', dist.Normal(jnp.zeros(a.shape[1]), 100))
    # Sample the outcome.
    numpyro.sample('y', dist.Normal(x * mo_weight + a * to_weight 
                                    + y_alpha[:, None] + y_beta[None, :], 
                                    NOISE_WEIGHT), obs=y)

def outcome_gm(a, x, y=None):
    """
    Array Geometric Model.
    """
    
    # Prior on the mediator-outcome weight.
    mo_weight = numpyro.sample('mo_weight', dist.Normal(0, 100))
    # Prior on the treatment-outcome weight.
    to_weight = numpyro.sample('to_weight', dist.Normal(0, 100))
    # Prior on the outcome latents.
    y_alpha = numpyro.sample('y_alpha', dist.Normal(jnp.zeros(a.shape[0]), 100))
    y_beta = numpyro.sample('y_beta', dist.Normal(jnp.zeros(a.shape[1]), 100))
    # Sample the outcome.
    numpyro.sample('y', dist.Normal(x * mo_weight + a * to_weight 
                                    + y_alpha[:, None] + y_beta[None, :], 
                                    NOISE_WEIGHT), obs=y)

def outcome_cm(a, x, y=None):
    """
    Conventional Causal Model.
    """
    
    # Prior on the treatment-mediator weight.
    tm_weight = numpyro.sample('tm_weight', dist.Normal(0, 100))
    # Sample the mediator.
    numpyro.sample('x', dist.Normal(a * tm_weight, 
                                    NOISE_WEIGHT), obs=x)
    

    # Prior on the mediator-outcome weight.
    mo_weight = numpyro.sample('mo_weight', dist.Normal(0, 100))
    # Prior on the treatment-outcome weight.
    to_weight = numpyro.sample('to_weight', dist.Normal(0, 100))
    # Sample the outcome.
    numpyro.sample('y', dist.Normal(x * mo_weight + a * to_weight, 
                                    NOISE_WEIGHT), obs=y)

def compute_treatment_effect(params, model_name, key):
    """
    Compute treatment effect from variational guide parameters.
    Assumes an autonormal guide. Computes the treatment effect and a 95% CI.
    """
    if model_name in ['outcome_gcm', 'outcome_cm']:
        # Monte Carlo estimate of the mean and CI.
        tm_samps =  (params['tm_weight_auto_loc'] + 
                     params['tm_weight_auto_scale'] * random.normal(key, (10000,)))
        mo_samps = (params['mo_weight_auto_loc'] + 
                    params['mo_weight_auto_scale'] * random.normal(key, (10000,)))
        effect_samps = tm_samps * mo_samps
        effect_mean = jnp.mean(effect_samps)
        effect_ci = jnp.percentile(effect_samps, jnp.array([2.5, 97.5]))
        return effect_mean, effect_ci
    elif model_name in ['outcome_gcm_misspec', 'outcome_gm']:
        effect_mean = params['to_weight_auto_loc']
        effect_cbar = 1.96 * params['to_weight_auto_scale']
        effect_ci = jnp.array([effect_mean - effect_cbar, effect_mean + effect_cbar])
        return effect_mean, effect_ci
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ============================================================================
# Run experiments
# ============================================================================
def run_single_experiment(experiment_id, domains, key, 
                          svi_num_steps=100000,
                          svi_step_size=0.0005,
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
    n_samples = max([domain[0] for domain in domains]
                    + [domain[1] for domain in domains])
    u, a, x, y, true_effect = dgp(n_samples, subkey)
    results['true_treatment_effect'] = float(true_effect)

    # Step 2: For each domain, subset data.
    for domain in domains:
        domain_results = {
            'domain': domain,
            'models': {}
        }
        
        # Subsample data to the domain
        u_sub, a_sub, x_sub, y_sub = subsample_data(u, a, x, y, domain)

        # Step 3: Run inference for each model.
        models = ['outcome_gcm', 'outcome_gm', 'outcome_cm'] if not only_gcm else ['outcome_gcm']
        for model_name in models:
            try:
                # Step 3: Run SVI.
                optimizer = numpyro.optim.Adam(step_size=svi_step_size)
                guide = AutoNormal(eval(model_name))
                svi = SVI(eval(model_name), guide, optimizer, loss=Trace_ELBO())
                key, subkey = random.split(key)
                svi_result = svi.run(subkey, svi_num_steps, a_sub, x_sub, y_sub)
                effect_mean, effect_ci = compute_treatment_effect(svi_result.params, model_name, subkey)
                # Compute statistics
                abs_error = float(np.abs(effect_mean - true_effect))
                covers_true = (effect_ci[0] <= true_effect <= effect_ci[1])
                
                domain_results['models'][model_name] = {
                    'posterior_mean': effect_mean,
                    'abs_error': abs_error,
                    'ci_25': effect_ci[0],
                    'ci_975': effect_ci[1],
                    'covers_true': covers_true,
                }
                print('Model: ', model_name, 'Posterior mean: ', effect_mean, 'True effect: ', true_effect)
                print('CI 2.5%: ', effect_ci[0], 'CI 97.5%: ', effect_ci[1])
                print('Absolute error: ', abs_error)
                print('Coverage: ', covers_true)

                if model_name == 'outcome_gcm':
                    # Compute effect estimates under misspecified graph.
                    key, subkey = random.split(key)
                    effect_mean, effect_ci = compute_treatment_effect(svi_result.params, 'outcome_gcm_misspec', subkey)
                    abs_error = float(np.abs(effect_mean - true_effect))
                    covers_true = (effect_ci[0] <= true_effect <= effect_ci[1])
                    domain_results['models'][model_name + '_misspec'] = {
                        'posterior_mean': effect_mean,
                        'abs_error': abs_error,
                        'ci_25': effect_ci[0],
                        'ci_975': effect_ci[1],
                        'covers_true': covers_true,
                    }
            except Exception as e:
                print(f"Error running {model_name} for domain {domain} in experiment {experiment_id}: {e}")
                domain_results['models'][model_name] = {
                    'error': str(e)
                }
        
        results['results_by_domain'].append(domain_results)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Run array GCM experiments')
    parser.add_argument('--n_experiments', type=int, default=10,
                        help='Number of experiments to run (default: 10)')
    parser.add_argument('--domains', type=str, default='(100,25);(200,50)',
                        help='Semicolon-separated list of domain tuples, default: (100,25);(200,50)')
    parser.add_argument('--output', type=str, default='array_gcm_results.pkl',
                        help='Output file path (default: array_gcm_results.pkl)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--svi_num_steps', type=int, default=100000,
                        help='Number of steps for SVI (default: 10000)')
    parser.add_argument('--svi_step_size', type=float, default=0.0005,
                        help='Step size for SVI (default: 0.0005)')
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
        domains.append((int(parts[0]), int(parts[1])))
    
    print(f"Running {args.n_experiments} experiments")
    print(f"Domains: {domains}")
    print(f"SVI: {args.svi_num_steps} steps, {args.svi_step_size} step size")
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
            exp_id, domains, subkey,
            args.svi_num_steps, args.svi_step_size,
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
            'domains': domains,
            'svi_num_steps': args.svi_num_steps,
            'svi_step_size': args.svi_step_size,
            'seed': args.seed
        }
    }
    
    with open(args.output, 'wb') as f:
        pickle.dump(output_data, f)
    
    print(f"Results saved to {args.output}")


if __name__ == '__main__':
    main()

"""
Experiment script for comparing CATE estimation methods on genomic track data.

Iterates over the cross-product of (scenario, noise, subset) and fits
naive, standard, and reparameterized models. Optionally fits non-RC-equivariant
variants of standard and reparam. Saves results (losses, true/estimated CATE,
true/estimated outcome) to a single .npz file for later plotting.

Usage examples
--------------
    python genomic_cate.py --scenarios 0 2 --noise 0.1 0.5 --subsets 200 500 --output results/out.npz
    python genomic_cate.py --scenarios 0 --noise 0.1 --subsets 200 --include_no_rc --repeats 5
    python genomic_cate.py --scenarios 0 --noise 0.1 --subsets 200 --num_steps 50000
    python genomic_cate.py --help
"""

import argparse
import itertools
import os
import time

import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO, Predictive
from jax.lax import conv_general_dilated, stop_gradient
from numpyro.infer.autoguide import AutoNormal

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WINDOW_GEN = 19
CONV_SIZE = 3
FEAT_SIZE = 1
WINDOW_OUTCOME = 11
WINDOW_PROPENSITY = 11
STEP_SIZE_NAIVE = 0.005
STEP_SIZE_STANDARD = 0.005
STEP_SIZE_REPARAM = 0.005
NUM_SVI_STEPS = 150_000
NUM_PREDICT_SAMPLES = 10
PRIOR_SCALE = 1.0
GENOME_REGION = os.path.join(os.path.dirname(__file__), "data", "tert.fa")

# ---------------------------------------------------------------------------
# Sequence utilities
# ---------------------------------------------------------------------------
base_to_idx = {"a": 0, "c": 1, "g": 2, "t": 3}

# Reverse complement of one-hot encoded nucleotides.
COMPLEMENT_MATRIX = jnp.array(
    [[0, 0, 0, 1], 
     [0, 0, 1, 0], 
     [0, 1, 0, 0], 
     [1, 0, 0, 0]]
)

# Reverse complement for the treatment effect vector.
TREATMENT_EFFECT_COMPLEMENT = jnp.array(
    [[-1, 0, 0], 
     [-1, 0, 1], 
     [-1, 1, 0]]
)


def load_fasta(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()
    header = lines[0].strip()[1:]
    sequence = "".join(line.strip() for line in lines[1:])
    return header, sequence


def reverse_complement(sequence):
    complement = {"a": "t", "t": "a", "c": "g", "g": "c"}
    return "".join(complement[base] for base in sequence[::-1])


def one_hot_encode(seq):
    indices = jnp.array([base_to_idx.get(b.lower(), 0) for b in seq])
    return jnp.eye(4)[indices]


# ---------------------------------------------------------------------------
# Data-generating process
# ---------------------------------------------------------------------------

def synthetic_outcome(seq, scenario=0):
    # True outcome.
    cent = int(np.floor(len(seq) / 2))
    if scenario == 0:
        return float(seq[cent - 1 : cent + 3] == "gtgc") * jnp.exp(
            1 + np.mean([float(l == "g" or l == "c") for l in list(seq[cent - 8 : cent + 9])])
        )
    elif scenario == 1:
        return float(seq[cent - 1 : cent + 2] == "gtg") * (
            1 + (seq[cent + 5 : cent + 8] in ["act", "tcg", "ttg"])
        )


def dgp(seq, key, scenario=0, noise=0.1):
    # Data generating process.
    seq_rc = reverse_complement(seq)
    key, subkey = random.split(key)
    noise_vec = np.array(random.normal(subkey, (len(seq),)) * noise)
    true_out = np.zeros((len(seq),))
    for j in range(WINDOW_GEN, len(seq) - WINDOW_GEN):
        true_out[j] += synthetic_outcome(
            seq[(j - (WINDOW_GEN // 2)) : (j + (WINDOW_GEN // 2) + 1)], scenario
        ) + synthetic_outcome(
            seq_rc[(-(j + 1) - WINDOW_GEN // 2) : (-(j + 1) + WINDOW_GEN // 2 + 1)],
            scenario,
        )
    return true_out + noise_vec, true_out


def true_cate(seq, scenario=0):
    # Calculate the true CATE at each position in the sequence.
    cate = {
        "a": np.zeros(len(seq)),
        "c": np.zeros(len(seq)),
        "g": np.zeros(len(seq)),
    }
    for j in range(WINDOW_GEN // 2, len(seq) - (WINDOW_GEN // 2)):
        segment = seq[j - (WINDOW_GEN // 2) : j + (WINDOW_GEN // 2) + 1]
        treat = segment[: (WINDOW_GEN // 2)] + "t" + segment[((WINDOW_GEN // 2) + 1) :]
        treat_out = synthetic_outcome(treat, scenario) + synthetic_outcome(
            reverse_complement(treat), scenario
        )
        for alt in ["a", "c", "g"]:
            ctrl = segment[: (WINDOW_GEN // 2)] + alt + segment[((WINDOW_GEN // 2) + 1) :]
            ctrl_out = synthetic_outcome(ctrl, scenario) + synthetic_outcome(
                reverse_complement(ctrl), scenario
            )
            cate[alt][j] = ctrl_out - treat_out
    return cate


# ---------------------------------------------------------------------------
# Neural-network building blocks
# ---------------------------------------------------------------------------

def _intervened_windows(seq, W, intervention=None):
    """Create sliding windows from a one-hot encoded sequence, optionally
    intervening at the center position of each window.

    Args:
        seq: one-hot encoded sequence, shape (L, 4)
        W: window size (number of positions per window)
        intervention: how to modify the center position of each window.
            None  - no modification.
            -1    - set center to all zeros.
            >= 0  - set center to one-hot at index `intervention`.

    Returns:
        windows: array of shape (L, W, 4)
    """
    seq_padded = np.pad(seq, ((W // 2, W // 2), (0, 0)), mode="constant")
    windows = np.moveaxis(
        np.lib.stride_tricks.sliding_window_view(seq_padded, W, 0), 1, 2
    )
    windows = jnp.array(windows)
    if intervention is not None:
        center = W // 2
        if intervention == -1:
            windows = windows.at[:, center, :].set(0.0)
        elif intervention >= 0:
            windows = windows.at[:, center, :].set(0.0)
            windows = windows.at[:, center, intervention].set(1.0)
    return windows


def nn(windows, filt, bias1, weights, bias2):
    """One-layer convolutional neural network.
    Args: 
        windows: one-hot encoded sequence formatted into sliding windows, L x W x 4
        filt: conv filter, K x 4 x F
        bias1: bias for conv layer, F
        weights: weights for average pooling, W x F X O
        bias2: weights for output, O

    Returns:
        out: output tensor, shape (L, O)

    Operations:
    - Convolve with filter, add bias, pass through softplus nonlinearity
    - Multiply by linear weights, add bias, output L x O tensor
    """
    L, W, _ = windows.shape
    K, _, F = filt.shape
    # Convolve filter along axis 1: result[i,j,f] = sum over (k,c) of windows[i, j-K//2:j+K//2+1, c] * filter[k, c, f]
    conv = conv_general_dilated(
        windows, filt, window_strides=(1,), padding="SAME",
        dimension_numbers=("NWC", "WIO", "NWC"),
    )
    # Nonlinearity.
    nonlin = jax.nn.softplus(conv + bias1)
    # Pad weights if smaller windows.
    if weights.shape[0] < W:
        weights = jnp.pad(weights, [((W - weights.shape[0])//2, (W - weights.shape[0])//2)] + [(0, 0)] * (len(weights.shape) - 1), mode="constant")
    # Multiply by linear weights and add bias.
    out = jnp.einsum("lwf,wf...->l...", nonlin, weights) + bias2
    return out


def outcome_nn(seq_windows, seq_rc_windows, filt, bias1, weights, bias2, rc_equiv=False):
    """Outcome model.
    Predicts scalar outcome at each position in the sequence.
    If rc_equiv=True, the model is reverse complement equivariant.
    Args:
        seq_windows: one-hot encoded sequence formatted into sliding windows, L x W x 4
        seq_rc_windows: reverse complement of seq_windows, L x W x 4
        filt: conv filter, K x 4 x F
        bias1: bias for conv layer, F
        weights: weights for average pooling, W x F
        bias2: weights for output, 1
        rc_equiv: whether to use reverse complement equivariant model

    Returns:
        out: output tensor, shape (L,)
    """
    fwd_out = nn(seq_windows, filt, bias1, weights, bias2)
    if not rc_equiv:
        return fwd_out
    rev_out = nn(seq_rc_windows, filt, bias1, weights, bias2)
    return fwd_out + rev_out[::-1]


def propensity_nn(seq_windows, seq_rc_windows, filt, bias1, weights, bias2, rc_equiv=False):
    """Propensity model.
    Predicts log probabilities for each nucleotide at each position in the sequence.
    If rc_equiv=True, the model is reverse complement equivariant.
    Args:
        seq_windows: one-hot encoded sequence formatted into sliding windows with zeros at the center position, L x W x 4
        seq_rc_windows: reverse complement of seq_windows with zeros at the center position, L x W x 4
        filt: conv filter, K x 4 x F
        bias1: bias for conv layer, F
        weights: weights for average pooling, W x F x 4
        bias2: bias for output, 4
        rc_equiv: whether to use reverse complement equivariant model

    Returns:
        out: output tensor, shape (L, 4)
    """
    fwd_logits = nn(seq_windows, filt, bias1, weights, bias2)
    fwd_softmax = jax.nn.log_softmax(fwd_logits)
    if not rc_equiv:
        return fwd_softmax
    rev_logits = nn(seq_rc_windows, filt, bias1, weights, bias2)
    rev_softmax = jax.nn.log_softmax(rev_logits)
    rev_softmax_rc = jnp.einsum("ij,lj->li", COMPLEMENT_MATRIX, rev_softmax[::-1])
    return jnp.logaddexp(fwd_softmax - jnp.log(2), rev_softmax_rc - jnp.log(2))


def treatment_effect_nn(seq_windows, seq_rc_windows, filt, bias1, weights, bias2, rc_equiv=False):
    """Treatment effect model.
    Predicts treatment effect at each position in the sequence.
    If rc_equiv=True, the model is reverse complement equivariant.
    Args:
        seq_windows: one-hot encoded sequence formatted into sliding windows with zeros at the center position, L x W x 4
        seq_rc_windows: reverse complement of seq_windows with zeros at the center position, L x W x 4
        filt: conv filter, K x 4 x F
        bias1: bias for conv layer, F
        weights: weights for average pooling, W x F x 3
        bias2: bias for output, 3
        rc_equiv: whether to use reverse complement equivariant model

    Returns:
        out: output tensor, shape (L, 3)
    """
    fwd_out = nn(seq_windows, filt, bias1, weights, bias2)
    if not rc_equiv:
        return fwd_out
    rev_out = nn(seq_rc_windows, filt, bias1, weights, bias2)
    rev_out_rc = jnp.einsum("ij,lj->li", TREATMENT_EFFECT_COMPLEMENT, rev_out[::-1])
    return 0.5 * fwd_out + 0.5 * rev_out_rc


# ---------------------------------------------------------------------------
# NumPyro models
# ---------------------------------------------------------------------------

def naive_model(seq, y=None):
    """Naive model, which does not account for interference across positions.
    Args:
        seq: one-hot encoded sequence, L x 4
        y: outcome, L
    """
    weights = numpyro.sample("weights", dist.Laplace(jnp.ones(4) * PRIOR_SCALE))
    bias = numpyro.sample("bias", dist.Normal(0, 100))
    y_mn = numpyro.deterministic("y_mn", jnp.einsum("ij,j->i", seq, weights) + bias)
    numpyro.sample("y", dist.Normal(y_mn, 0.1), obs=y)


def naive_te_estimate(seq, weights):
    """Compute treatment effect for the naive model.
    Args:
        seq: one-hot encoded sequence, L x 4
        weights: posterior mean weights for the naive model, 4

    Returns:
        te: treatment effect, L x 3
    """
    return (weights[:-1] - weights[-1]) * jnp.ones((seq.shape[0], 1))


def standard_model(seq_windows, seq_rc_windows, y=None, rc_equiv=False):
    """Standard genomic treatment effect modeling, using just an outcome model
    Args:
        seq_windows: sequence formatted as sliding windows, L x W x 4
        seq_rc_windows: reverse complement of seq_windows, L x W x 4
        y: outcome, L
        rc_equiv: whether to use reverse complement equivariant model
    """
    filt = numpyro.sample("filter", dist.Laplace(scale=jnp.ones((CONV_SIZE, 4, FEAT_SIZE)) * PRIOR_SCALE))
    bias1 = numpyro.sample("bias1", dist.Normal(jnp.zeros(FEAT_SIZE), 100))
    weights = numpyro.sample("weights", dist.Laplace(scale=jnp.ones((WINDOW_OUTCOME, FEAT_SIZE)) * PRIOR_SCALE))
    bias2 = numpyro.sample("bias2", dist.Normal(0, 100))
    y_mn = numpyro.deterministic(
        "y_mn",
        outcome_nn(seq_windows, seq_rc_windows, filt, bias1, weights, bias2, rc_equiv=rc_equiv),
    )
    numpyro.sample("y", dist.Normal(y_mn, 0.1), obs=y)


def standard_te_via_intervention(seq_oh, seq_rc_oh, predictive_fn, subkey, rc_equiv):
    """Compute treatment effect for the standard model.
    Args:
        seq_oh: one-hot encoded sequence, L x 4
        seq_rc_oh: reverse complement of seq_oh, L x 4
        predictive_fn: numpyro predictive, taking a key and a sequence formatted as sliding windows
        subkey: random key
        rc_equiv: whether using a reverse complement equivariant model

    Returns:
        te: treatment effect, L x 3
    """
    intervene_ymn = jnp.concat(
        [
            predictive_fn(
                subkey,
                _intervened_windows(seq_oh, WINDOW_GEN, li),
                _intervened_windows(seq_rc_oh, WINDOW_GEN, int(jnp.argmax(COMPLEMENT_MATRIX[li]))),
                rc_equiv=rc_equiv,
            )["y_mn"].mean(axis=0)[:, None]
            for li in range(4)
        ],
        axis=1,
    )
    return intervene_ymn[:, :-1] - intervene_ymn[:, -1:]


def reparam_model(seq_windows_0, seq_rc_windows_0, seq, y=None, rc_equiv=False):
    """Reparameterized treatment effect model, using Neyman orthogonality.
    Args:
        seq_windows_0: sequence formatted as sliding windows with zeros at the center position, L x W x 4
        seq_rc_windows_0: reverse complement of seq_windows_0 with zeros at the center position, L x W x 4
        seq: one-hot encoded sequence (forward strand), L x 4
        y: outcome, L
        rc_equiv: whether to use reverse complement equivariant model
    """
    prop_filt = numpyro.sample("prop_filter", dist.Laplace(jnp.ones((CONV_SIZE, 4, FEAT_SIZE)) * PRIOR_SCALE))
    prop_bias1 = numpyro.sample("prop_bias1", dist.Normal(jnp.zeros(FEAT_SIZE), 100))
    prop_weights = numpyro.sample("prop_weights", dist.Laplace(jnp.ones((WINDOW_PROPENSITY, FEAT_SIZE, 4)) * PRIOR_SCALE))
    prop_bias2 = numpyro.sample("prop_bias2", dist.Normal(jnp.zeros(4), 100))
    prop = numpyro.deterministic("prop", propensity_nn(seq_windows_0, seq_rc_windows_0, prop_filt, prop_bias1, prop_weights, prop_bias2, rc_equiv=rc_equiv)) # Equivariant propensity. 
    numpyro.sample("seq", dist.MultinomialLogits(logits=prop), obs=seq)

    out_filt = numpyro.sample("out_filter", dist.Laplace(jnp.ones((CONV_SIZE, 4, FEAT_SIZE)) * PRIOR_SCALE))
    out_bias1 = numpyro.sample("out_bias1", dist.Normal(jnp.zeros(FEAT_SIZE), 100))
    out_weights = numpyro.sample("out_weights", dist.Laplace(jnp.ones((WINDOW_OUTCOME, FEAT_SIZE)) * PRIOR_SCALE))
    out_bias2 = numpyro.sample("out_bias2", dist.Normal(0, 100))
    out_mn = numpyro.deterministic(
        "out_mn",
        outcome_nn(seq_windows_0, seq_rc_windows_0, out_filt, out_bias1, out_weights, out_bias2, rc_equiv=rc_equiv),  # Equivariant outcome. 
    )

    te_filt = numpyro.sample("te_filter", dist.Laplace(jnp.ones((CONV_SIZE, 4, FEAT_SIZE)) * PRIOR_SCALE))
    te_bias1 = numpyro.sample("te_bias1", dist.Normal(jnp.zeros(FEAT_SIZE), 100))
    te_weights = numpyro.sample("te_weights", dist.Laplace(jnp.ones((WINDOW_OUTCOME, FEAT_SIZE, 3)) * PRIOR_SCALE))
    te_bias2 = numpyro.sample("te_bias2", dist.Normal(jnp.zeros(3), 100))
    te = numpyro.deterministic(
        "te",
        treatment_effect_nn(seq_windows_0, seq_rc_windows_0, te_filt, te_bias1, te_weights, te_bias2, rc_equiv=False),
    )

    y_mn = numpyro.deterministic(
        "y_mn",
        out_mn + jnp.einsum("lk,lk->l", seq[:, :-1] - stop_gradient(prop)[:, :-1], te),
    )
    numpyro.sample("y", dist.Normal(y_mn, 0.1), obs=y)



# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _fit_naive(key, seq_oh, outcome_sub, num_steps=NUM_SVI_STEPS):
    optimizer = numpyro.optim.Adam(step_size=STEP_SIZE_NAIVE)
    guide = AutoNormal(naive_model)
    svi = SVI(naive_model, guide, optimizer, loss=Trace_ELBO())
    key, subkey = random.split(key)
    result = svi.run(subkey, num_steps, seq_oh, outcome_sub, progress_bar=False)
    return result, guide, key


def _fit_standard(key, seq_windows, seq_rc_windows, outcome_sub, rc_equiv, num_steps=NUM_SVI_STEPS):
    optimizer = numpyro.optim.Adam(step_size=STEP_SIZE_STANDARD)
    guide = AutoNormal(standard_model)
    svi = SVI(standard_model, guide, optimizer, loss=Trace_ELBO())
    key, subkey = random.split(key)
    result = svi.run(subkey, num_steps, seq_windows, seq_rc_windows, outcome_sub, rc_equiv=rc_equiv, progress_bar=False)
    return result, guide, key


def _fit_reparam(key, seq_windows_0, seq_rc_windows_0, seq_oh, outcome_sub, rc_equiv, num_steps=NUM_SVI_STEPS):
    optimizer = numpyro.optim.Adam(step_size=STEP_SIZE_REPARAM)
    guide = AutoNormal(reparam_model)
    svi = SVI(reparam_model, guide, optimizer, loss=Trace_ELBO())
    key, subkey = random.split(key)
    result = svi.run(subkey, num_steps, seq_windows_0, seq_rc_windows_0, seq_oh, outcome_sub, rc_equiv=rc_equiv, progress_bar=False)
    return result, guide, key


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _eval_naive(key, result, guide, seq_oh, outcome_sub, true_out_sub, cate_dict, subset):
    predictive = Predictive(naive_model, guide=guide, params=result.params, num_samples=NUM_PREDICT_SAMPLES)
    key, subkey = random.split(key)
    samps = predictive(subkey, seq_oh)
    y_mn = np.array(samps["y_mn"].mean(axis=0))
    pred_mse = float(((samps["y_mn"].mean(axis=0) - true_out_sub) ** 2).mean())
    te = naive_te_estimate(seq_oh, result.params["weights_auto_loc"])
    te_mse = float(jnp.mean(jnp.concat([(te[:, li] - cate_dict[l][:subset]) ** 2 for li, l in enumerate(["a", "c", "g"])])))
    return {
        "losses": np.array(result.losses),
        "pred_mse": pred_mse,
        "te_mse": te_mse,
        "y_mn": y_mn,
        "te_est": np.array(te),
    }, key


def _eval_standard(key, result, guide, seq_oh, seq_rc_oh, seq_windows, seq_rc_windows, outcome_sub, true_out_sub, cate_dict, subset, rc_equiv):
    predictive = Predictive(standard_model, guide=guide, params=result.params, num_samples=NUM_PREDICT_SAMPLES)
    key, subkey = random.split(key)
    samps = predictive(subkey, seq_windows, seq_rc_windows, rc_equiv=rc_equiv)
    y_mn = np.array(samps["y_mn"].mean(axis=0))
    pred_mse = float(((samps["y_mn"].mean(axis=0) - true_out_sub) ** 2).mean())
    te = standard_te_via_intervention(seq_oh, seq_rc_oh, predictive, subkey, rc_equiv)
    te_mse = float(jnp.mean(jnp.concat([(te[:, li] - cate_dict[l][:subset]) ** 2 for li, l in enumerate(["a", "c", "g"])])))
    return {
        "losses": np.array(result.losses),
        "pred_mse": pred_mse,
        "te_mse": te_mse,
        "y_mn": y_mn,
        "te_est": np.array(te),
    }, key


def _eval_reparam(key, result, guide, seq_windows_0, seq_rc_windows_0, seq_oh, outcome_sub, true_out_sub, cate_dict, subset, rc_equiv):
    predictive = Predictive(reparam_model, guide=guide, params=result.params, num_samples=NUM_PREDICT_SAMPLES)
    key, subkey = random.split(key)
    samps = predictive(subkey, seq_windows_0, seq_rc_windows_0, seq_oh, rc_equiv=rc_equiv)
    y_mn = np.array(samps["y_mn"].mean(axis=0))
    pred_mse = float(((samps["y_mn"].mean(axis=0) - true_out_sub) ** 2).mean())
    te = np.array(samps["te"].mean(axis=0))
    te_mse = float(jnp.mean(jnp.concat([(te[:, li] - cate_dict[l][:subset]) ** 2 for li, l in enumerate(["a", "c", "g"])])))
    prop_ll = jnp.mean(jnp.sum(samps["prop"].mean(axis=0) * seq_oh, axis=1))
    return {
        "losses": np.array(result.losses),
        "pred_mse": pred_mse,
        "te_mse": te_mse,
        "y_mn": y_mn,
        "te_est": te,
        "prop_perplexity": float(jnp.exp(-prop_ll)),
    }, key


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run_experiments(scenarios, noise_levels, subsets, include_no_rc, seed=3,
                    repeats=1, num_steps=None, output="results/genomic_cate_results.npz",
                    args_dict=None):
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    _, genome_sequence = load_fasta(GENOME_REGION)
    all_results = {}
    steps = num_steps or NUM_SVI_STEPS

    for rep in range(repeats):
        rep_seed = seed + rep
        for scenario, noise, subset in itertools.product(scenarios, noise_levels, subsets):
            condition_key = f"repeat={rep}_scenario={scenario}_noise={noise}_subset={subset}"
            print(f"\n{'='*60}")
            print(f"  {condition_key}")
            print(f"{'='*60}")

            key = random.key(rep_seed)
            outcome, true_out = dgp(genome_sequence, key, scenario=scenario, noise=noise)
            cate_dict = true_cate(genome_sequence, scenario=scenario)
            outcome_sub = outcome[:subset]
            true_out_sub = true_out[:subset]

            cate_mat = np.column_stack([cate_dict["a"][:subset], cate_dict["c"][:subset], cate_dict["g"][:subset]])

            seq_oh = one_hot_encode(genome_sequence[:subset])
            seq_rc_oh = one_hot_encode(reverse_complement(genome_sequence[:subset]))
            window_max = max(WINDOW_OUTCOME, WINDOW_PROPENSITY)
            seq_windows = _intervened_windows(seq_oh, window_max)
            seq_rc_windows = _intervened_windows(seq_rc_oh, window_max)
            seq_windows_0 = _intervened_windows(seq_oh, window_max, -1)
            seq_rc_windows_0 = _intervened_windows(seq_rc_oh, window_max, -1)

            condition_results = {
                "outcome": np.array(outcome_sub),
                "true_outcome": np.array(true_out_sub),
                "true_cate": cate_mat,
            }

            # --- Naive ---
            print("  Fitting naive model ...")
            t0 = time.time()
            res, guide, key = _fit_naive(key, seq_oh, outcome_sub, num_steps=steps)
            metrics, key = _eval_naive(key, res, guide, seq_oh, outcome_sub, true_out_sub, cate_dict, subset)
            print(f"    done in {time.time()-t0:.1f}s | pred_mse={metrics['pred_mse']:.6f} | te_mse={metrics['te_mse']:.6f}")
            condition_results["naive"] = metrics

            # --- Standard (rc_equiv=True) ---
            print("  Fitting standard model (rc_equiv=True) ...")
            t0 = time.time()
            res, guide, key = _fit_standard(key, seq_windows, seq_rc_windows, outcome_sub, rc_equiv=True, num_steps=steps)
            metrics, key = _eval_standard(key, res, guide, seq_oh, seq_rc_oh, seq_windows, seq_rc_windows, outcome_sub, true_out_sub, cate_dict, subset, rc_equiv=True)
            print(f"    done in {time.time()-t0:.1f}s | pred_mse={metrics['pred_mse']:.6f} | te_mse={metrics['te_mse']:.6f}")
            condition_results["standard_rc"] = metrics

            # --- Standard (rc_equiv=False) ---
            if include_no_rc:
                print("  Fitting standard model (rc_equiv=False) ...")
                t0 = time.time()
                res, guide, key = _fit_standard(key, seq_windows, seq_rc_windows, outcome_sub, rc_equiv=False, num_steps=steps)
                metrics, key = _eval_standard(key, res, guide, seq_oh, seq_rc_oh, seq_windows, seq_rc_windows, outcome_sub, true_out_sub, cate_dict, subset, rc_equiv=False)
                print(f"    done in {time.time()-t0:.1f}s | pred_mse={metrics['pred_mse']:.6f} | te_mse={metrics['te_mse']:.6f}")
                condition_results["standard_no_rc"] = metrics

            # --- Reparam (rc_equiv=True) ---
            print("  Fitting reparam model (rc_equiv=True) ...")
            t0 = time.time()
            res, guide, key = _fit_reparam(key, seq_windows_0, seq_rc_windows_0, seq_oh, outcome_sub, rc_equiv=True, num_steps=steps)
            metrics, key = _eval_reparam(key, res, guide, seq_windows_0, seq_rc_windows_0, seq_oh, outcome_sub, true_out_sub, cate_dict, subset, rc_equiv=True)
            print(f"    done in {time.time()-t0:.1f}s | pred_mse={metrics['pred_mse']:.6f} | te_mse={metrics['te_mse']:.6f}")
            condition_results["reparam_rc"] = metrics

            # --- Reparam (rc_equiv=False) ---
            if include_no_rc:
                print("  Fitting reparam model (rc_equiv=False) ...")
                t0 = time.time()
                res, guide, key = _fit_reparam(key, seq_windows_0, seq_rc_windows_0, seq_oh, outcome_sub, rc_equiv=False, num_steps=steps)
                metrics, key = _eval_reparam(key, res, guide, seq_windows_0, seq_rc_windows_0, seq_oh, outcome_sub, true_out_sub, cate_dict, subset, rc_equiv=False)
                print(f"    done in {time.time()-t0:.1f}s | pred_mse={metrics['pred_mse']:.6f} | te_mse={metrics['te_mse']:.6f}")
                condition_results["reparam_no_rc"] = metrics

            all_results[condition_key] = condition_results

    # Save everything as a single .npz with nested dict structure serialised
    # via a flat key scheme:  "condition_key/model_name/field"
    flat = {}
    if args_dict is not None:
        for arg_name, arg_val in args_dict.items():
            flat[f"args/{arg_name}"] = np.asarray(arg_val)
    for cond_key, cond_dict in all_results.items():
        for field_or_model, value in cond_dict.items():
            if isinstance(value, dict):
                for metric_name, metric_val in value.items():
                    flat[f"{cond_key}/{field_or_model}/{metric_name}"] = np.asarray(metric_val)
            else:
                flat[f"{cond_key}/{field_or_model}"] = np.asarray(value)

    np.savez(output, **flat)
    print(f"\nResults saved to {output}")
    return all_results


# ---------------------------------------------------------------------------
# Command line interface
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run CATE estimation experiments on genomic data.")
    parser.add_argument("--scenarios", type=int, nargs="+", default=[0], help="DGP scenarios to test (0-3)")
    parser.add_argument("--noise", type=float, nargs="+", default=[0.1], help="Noise levels")
    parser.add_argument("--subsets", type=int, nargs="+", default=[200], help="Subset sizes of the genome to use")
    parser.add_argument("--include_no_rc", action="store_true", help="Also fit non-RC-equivariant standard and reparam models")
    parser.add_argument("--seed", type=int, default=3, help="Random seed")
    parser.add_argument("--repeats", type=int, default=1, help="Number of times to repeat each experiment (seed increments per repeat)")
    parser.add_argument("--num_steps", type=int, default=None, help="Number of SVI steps (default: %(default)s uses built-in constants)")
    parser.add_argument("--output", type=str, default="results/genomic_cate_results.npz", help="Output .npz file path")
    args = parser.parse_args()

    args_dict = {
        "scenarios": args.scenarios,
        "noise": args.noise,
        "subsets": args.subsets,
        "include_no_rc": args.include_no_rc,
        "seed": args.seed,
        "repeats": args.repeats,
        "num_steps": args.num_steps if args.num_steps is not None else NUM_SVI_STEPS,
    }

    run_experiments(
        scenarios=args.scenarios,
        noise_levels=args.noise,
        subsets=args.subsets,
        include_no_rc=args.include_no_rc,
        seed=args.seed,
        repeats=args.repeats,
        num_steps=args.num_steps,
        output=args.output,
        args_dict=args_dict,
    )


if __name__ == "__main__":
    main()

import numpy as np

# Based on https://github.com/gostevehoward/confseq

def lambda_predmix_eb(
    x,
    truncation=np.inf,               # safer default than ∞
    alpha=0.05,
    fixed_n=None,
    prior_mean=1 / 2,
    prior_variance=1 / 4,
    fake_obs=1,
    scale=1.,
    A=0,
    B=1,
    verbose=False
):
    t = np.arange(1, len(x) + 1)
    mu_hat_t = np.minimum((fake_obs * prior_mean + np.cumsum(x)) / (t + fake_obs), 1)
    sigma2_t = (fake_obs * prior_variance + np.cumsum(np.power(x - mu_hat_t, 2))) / (t + fake_obs)
    sigma2_tminus1 = np.append(prior_variance, sigma2_t[0: (len(x) - 1)])

    lambdas = np.minimum(np.sqrt(2 * np.log(1 / alpha) / (fixed_n * sigma2_tminus1)), 1/(B-A))
    lambdas[np.isnan(lambdas)] = 0
    lambdas = np.minimum(truncation, lambdas)

    if verbose:
        print(sigma2_t)
        print("===")
        print(lambdas)

    return lambdas * scale


def betting_mart(losses, alpha, lambda_fn_positive):
    n = len(losses)
    martingale_pos = np.ones(n)

    lambda_positive_vals = lambda_fn_positive(losses, n)

    for t in range(1, n):
        nu_t = lambda_positive_vals[t]
        term = 1 - nu_t * (losses[t] - alpha)
        martingale_pos[t] = martingale_pos[t - 1] * term

    return martingale_pos


def wsr_p_value(losses, alpha=0.1, delta=0.1, A= 0, B=1, scale=1., verbose=False):
    n = len(losses)

    lambda_fn = lambda x, m: lambda_predmix_eb(
        x,
        alpha=delta,
        fixed_n=n,
        scale=scale,
        A=A, B=B,
        verbose=verbose
    )

    n = len(losses)
    capital_neg = np.ones(n)

    lambdas = lambda_fn(losses, n)

    for t in range(1, n):
        lam = lambdas[t]
        multiplicand = 1 - lam * (losses[t] - alpha)
        capital_neg[t] = capital_neg[t - 1] * multiplicand

    pval = min(1.0, 1 / np.max(capital_neg))

    return pval
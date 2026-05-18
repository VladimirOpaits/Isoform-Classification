import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

SPACE = {
    'n_estimators':      ('int',   100,  2000),
    'max_depth':         ('int',   5,    50),
    'min_samples_split': ('int',   2,    30),
    'min_samples_leaf':  ('int',   1,    15),
    'max_features':      ('float', 0.1,  1.0),
    'max_samples':       ('float', 0.5,  1.0),
}

def config_to_vector(config):
    vec = []
    for name, spec in SPACE.items():
        kind = spec[0]
        lo, hi = spec[1], spec[2]
        vec.append((config[name] - lo) / (hi - lo))
    return np.array(vec)

def random_config(rng):
    config = {}
    for name, spec in SPACE.items():
        kind = spec[0]
        if kind == 'int':
            config[name] = int(rng.integers(spec[1], spec[2] + 1))
        elif kind == 'float':
            config[name] = float(rng.uniform(spec[1], spec[2]))
    return config

def expected_improvement(X_cand, gp, y_best, xi=0.01):
    mu, sigma = gp.predict(X_cand, return_std=True)
    z = (mu - y_best - xi) / (sigma + 1e-9)
    ei = (mu - y_best - xi) * norm.cdf(z) + sigma * norm.pdf(z)
    ei[sigma < 1e-9] = 0.0
    return ei

def suggest_next(X_obs, y_obs, rng, n_candidates=1000):
    gp = GaussianProcessRegressor(
        kernel=Matern(nu=2.5),
        n_restarts_optimizer=5,
        normalize_y=True,
    )
    gp.fit(X_obs, y_obs)
    candidates = [random_config(rng) for _ in range(n_candidates)]
    X_cand = np.array([config_to_vector(c) for c in candidates])
    ei = expected_improvement(X_cand, gp, y_best=np.max(y_obs))
    return candidates[np.argmax(ei)], gp

def suggest_batch(X_obs, y_obs, rng, n, n_candidates=1000):
    X_aug, y_aug = X_obs.copy(), y_obs.copy()
    batch = []
    for _ in range(n):
        config, gp = suggest_next(X_aug, y_aug, rng, n_candidates)
        batch.append(config)
        vec = config_to_vector(config)
        y_hal = gp.predict([vec])[0]
        X_aug = np.vstack([X_aug, vec])
        y_aug = np.append(y_aug, y_hal)
    return batch
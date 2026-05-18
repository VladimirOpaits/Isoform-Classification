import argparse
import json
import sys
import os
import warnings
import numpy as np
from mpi4py import MPI
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings('ignore', category=ConvergenceWarning)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from settings import LABELED_PATH, BEST_PARAMS_PATH
from training.bayes_opt import SPACE, suggest_batch, random_config, config_to_vector


def load_labeled(path):
    df = pd.read_parquet(path)
    X = df.drop(columns=['label']).values.astype(np.float32)
    y = df['label'].values
    return X, y


def master(comm, n_calls, data_path, output_path, seed):
    n_workers = comm.Get_size() - 1
    rng = np.random.default_rng(seed)
    n_rounds = (n_calls + n_workers - 1) // n_workers

    X_obs = np.empty((0, len(SPACE)))
    y_obs = np.empty(0)
    best_score = -np.inf
    best_config = None

    for round_idx in range(n_rounds):
        if round_idx == 0:
            batch = [random_config(rng) for _ in range(n_workers)]
        else:
            batch = suggest_batch(X_obs, y_obs, rng, n=n_workers)

        for rank, config in enumerate(batch, start=1):
            comm.send(config, dest=rank, tag=0)

        for _ in range(n_workers):
            result = comm.recv(source=MPI.ANY_SOURCE, tag=1)
            X_obs = np.vstack([X_obs, config_to_vector(result['config'])])
            y_obs = np.append(y_obs, result['oob'])
            if result['oob'] > best_score:
                best_score = result['oob']
                best_config = result['config']

        print(f"round {round_idx + 1}/{n_rounds}  best_oob={best_score:.4f}", flush=True)
        with open(output_path, 'w') as f:
            json.dump({'config': best_config, 'oob_score': best_score}, f, indent=2)

    for rank in range(1, n_workers + 1):
        comm.send(None, dest=rank, tag=0)


def worker(comm, data_path):
    X, y = load_labeled(data_path)
    while True:
        config = comm.recv(source=0, tag=0)
        if config is None:
            break
        clf = RandomForestClassifier(oob_score=True, n_jobs=1, **config)
        clf.fit(X, y)
        comm.send({'config': config, 'oob': clf.oob_score_}, dest=0, tag=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-calls', type=int, default=64)
    parser.add_argument('--data', default=LABELED_PATH)
    parser.add_argument('--output', default=BEST_PARAMS_PATH)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        master(comm, args.n_calls, args.data, args.output, args.seed)
    else:
        worker(comm, args.data)


if __name__ == '__main__':
    main()

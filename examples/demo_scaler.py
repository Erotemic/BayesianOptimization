from bayes_opt import BayesianOptimization
from bayes_opt.helpers import ensure_rng
import numpy as np


def test_scalar():
    # Create domain with very different scales
    x1s = np.linspace(-2, 10, 10)
    x2s = np.linspace(-2, 10, 1000)
    x3s = np.linspace(-2, 10, 100000)

    def f(xs):
        return np.exp(-(xs - 2)**2) + np.exp(-(xs - 6)**2 / 10) + 1 / (xs**2 + 1)

    f1 = f(x1s)
    f2 = f(x2s)
    f3 = f(x3s)

    def target_func(x1, x2, x3):
        return f1[int(x1)] + f2[int(x2)] + f3[int(x3)]

    pbounds = {
        'x1': (0, len(f1) - 1),
        'x2': (0, len(f2) - 1),
        'x3': (0, len(f3) - 1),
    }

    gp_params = {'alpha': 1e-5, 'n_restarts_optimizer': 2}

    # First test with scaler=False
    print('--- WITHOUT SCALER ---')
    random_state = ensure_rng(0)
    bo = BayesianOptimization(f=target_func,
                              pbounds=pbounds,
                              random_state=random_state,
                              verbose=1)
    # Change aquisition params to speedup optimization for testing purposes
    bo.maximize(init_points=10, n_iter=25, acq='ucb', kappa=5, scaler=False,
                **gp_params)
    res = bo.space.max_point()
    max_val = res['max_val']
    print('max_val = {!r}'.format(max_val))

    print('--- WITH SCALER ---')
    # Now test with scaler = True
    random_state = ensure_rng(0)
    bo2 = BayesianOptimization(f=target_func,
                               pbounds=pbounds,
                               random_state=random_state,
                               verbose=1)
    # Change aquisition params to speedup optimization for testing purposes
    bo2.maximize(init_points=10, n_iter=25, acq='ucb', kappa=5, scaler=True,
                 **gp_params)
    res = bo2.space.max_point()
    max_val2 = res['max_val']
    print('max_val2 = {!r}'.format(max_val2))


if __name__ == '__main__':
    r"""
    CommandLine:
        python examples/demo_scaler.py
    """
    test_scalar()

import numpy as np
from lvproxy.mlp import Arch, ACTIVATIONS, init_params, loss_and_grads


def test_gradcheck():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(7, 5))
    y = rng.integers(0, 3, size=7)
    for act in ACTIVATIONS:
        for res in (False, True):
            arch = Arch(depth=2, width=4, act=act, residual=res, gain=1.3)
            p = init_params(arch, 5, 3, np.random.default_rng(1))
            _, g = loss_and_grads(arch, p, X, y)
            for li in range(len(p)):
                for j in range(2):
                    idx = tuple(rng.integers(0, s) for s in p[li][j].shape)
                    old = p[li][j][idx]
                    p[li][j][idx] = old + 1e-6
                    lp, _ = loss_and_grads(arch, p, X, y)
                    p[li][j][idx] = old - 1e-6
                    lm, _ = loss_and_grads(arch, p, X, y)
                    p[li][j][idx] = old
                    num = (lp - lm) / 2e-6
                    assert abs(num - g[li][j][idx]) < 1e-5 * max(1, abs(num)), (act, res, li, j)


if __name__ == "__main__":
    test_gradcheck()
    print("gradcheck ok")

import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
from sklearn import metrics

def get_samples(m_x, s_x, m_w, s_w, N):

    v_x = s_x ** 2
    v_w = s_w ** 2
    x = np.random.normal(size=N) * s_x + m_x
    w = np.random.normal(size=N) * s_w + m_w

    t = np.maximum(x * w, 0)
    x2 = np.random.normal(size=N) * s_x + m_x
    w2 = np.random.normal(size=N) * s_w + m_w
    t2 = np.maximum(x2 * w2, 0)


    pbp_lin_m = m_w * m_x
    pbp_lin_v = v_w * v_x + m_w ** 2 * v_x + v_w * m_x ** 2

    pbp_alpha = pbp_lin_m / np.sqrt(pbp_lin_v)
    pbp_gamma = ss.norm.pdf(pbp_alpha) / ss.norm.cdf(pbp_alpha)
    pbp_v_prime = pbp_lin_m + np.sqrt(pbp_lin_v) * pbp_gamma

    pbp_relu_m = ss.norm.cdf(pbp_alpha) * pbp_v_prime
    pbp_relu_v = (pbp_relu_m * pbp_v_prime * ss.norm.cdf(-pbp_alpha) + 
                ss.norm.cdf(pbp_alpha) * pbp_lin_v * (1 - pbp_gamma * (pbp_gamma + pbp_alpha)))
    pbp_sample = np.random.normal(size=N) * np.sqrt(pbp_relu_v) + pbp_relu_m

    pi_l = 1
    pi = 1
    sspbp_lin_m = m_w * m_x
    sspbp_lin_v = (
        (
            v_w * v_x + m_w ** 2 * v_x + v_w * m_x ** 2 + 
            m_w ** 2 * pi * (1 - pi) * m_x ** 2
        ) / pi_l - (1 - pi_l) * sspbp_lin_m ** 2
    )
    sspbp_lin_rho = pi_l

    sspbp_alpha = sspbp_lin_m / np.sqrt(sspbp_lin_v)
    sspbp_gamma = ss.norm.pdf(sspbp_alpha) / ss.norm.cdf(sspbp_alpha)

    sspbp_relu_rho = sspbp_lin_rho * ss.norm.cdf(sspbp_alpha)
    sspbp_relu_m = sspbp_lin_m + np.sqrt(sspbp_lin_v) * sspbp_gamma
    sspbp_relu_v = sspbp_lin_v * (1 - sspbp_gamma * (sspbp_gamma + sspbp_alpha))

    sspbp_relu_z = np.random.choice([0, 1], size=N, p=[1 - sspbp_relu_rho, sspbp_relu_rho])
    sspbp_sample = (1 - sspbp_relu_z) * 0 + sspbp_relu_z * (
        np.random.normal(size=N) * np.sqrt(sspbp_relu_v) + sspbp_relu_m)
    return t, t2, pbp_sample, sspbp_sample

def mmd_rbf(X, Y, gamma=1.0):
    X = X.reshape((-1, 1))
    Y = Y.reshape((-1, 1))
    XX = metrics.pairwise.rbf_kernel(X, X, gamma)
    YY = metrics.pairwise.rbf_kernel(Y, Y, gamma)
    XY = metrics.pairwise.rbf_kernel(X, Y, gamma)
    return XX.mean() + YY.mean() - 2 * XY.mean()

def compare(m_x, s_x, m_w, s_w, N):
    t, t2, pbp_sample, sspbp_sample = get_samples(m_x, s_x, m_w, s_w, N)
    same, pbp, sspbp = (mmd_rbf(t, t2), mmd_rbf(t, pbp_sample), mmd_rbf(t, sspbp_sample))
    print("$N(%d, %d)$ & $N(%d, %d)$ & %.2f\\%% & %0.9f & %0.9f & %0.9f \\\\" % (m_x, s_x**2, m_w, s_w **2, np.mean(t <= 0) * 100, same, pbp, sspbp))
#     return same, pbp, sspbp

print("$p_X$ & $p_W$ & \\% Saturated & Same & PBP & SSPBP \\\\")


N = 10000
compare(0, 1, 0, 1, N)
compare(1, 1, 3, 1, N)
compare(1, 1, -3, 1, N)
compare(3, 1, 3, 1, N)
compare(3, 1, -3, 1, N)

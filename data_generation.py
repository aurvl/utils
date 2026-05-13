import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset


# ══════════════════════════════════════════════════════
# CATÉGORIE 1 — OSCILLATEURS & PÉRIODIQUES (15)
# ══════════════════════════════════════════════════════

def gen_sine(n, noise=0.05):
    t = np.linspace(0, 4*np.pi, n)
    f = np.random.uniform(0.5, 3.0)
    p = np.random.uniform(0, 2*np.pi)
    a = np.random.uniform(0.5, 2.0)
    return a * np.sin(f*t + p) + np.random.randn(n)*noise

def gen_cosine(n, noise=0.05):
    t = np.linspace(0, 4*np.pi, n)
    f = np.random.uniform(0.5, 3.0)
    a = np.random.uniform(0.5, 2.0)
    return a * np.cos(f*t) + np.random.randn(n)*noise

def gen_square_wave(n, noise=0.05):
    t = np.linspace(0, 4*np.pi, n)
    f = np.random.uniform(0.5, 2.0)
    x = np.sign(np.sin(f*t))
    return x + np.random.randn(n)*noise

def gen_sawtooth(n, noise=0.05):
    t = np.linspace(0, 1, n)
    f = np.random.randint(3, 10)
    x = 2*(t*f - np.floor(0.5 + t*f))
    return x + np.random.randn(n)*noise

def gen_triangle_wave(n, noise=0.05):
    t = np.linspace(0, 1, n)
    f = np.random.randint(3, 8)
    x = 2*np.abs(2*(t*f - np.floor(t*f + 0.5))) - 1
    return x + np.random.randn(n)*noise

def gen_multi_sine(n, noise=0.05):
    """Superposition de 3-6 sinus — simule multi-saisonnalité"""
    k = np.random.randint(3, 7)
    x = sum(gen_sine(n, noise=0.0) for _ in range(k)) / k
    return x + np.random.randn(n)*noise

def gen_am_signal(n, noise=0.05):
    """Amplitude Modulated — enveloppe qui pulse"""
    t = np.linspace(0, 4*np.pi, n)
    carrier = np.sin(np.random.uniform(2,5)*t)
    envelope = 1 + 0.5*np.sin(np.random.uniform(0.1, 0.5)*t)
    return carrier * envelope + np.random.randn(n)*noise

def gen_fm_signal(n, noise=0.05):
    """Frequency Modulated — fréquence qui varie"""
    t = np.linspace(0, 4*np.pi, n)
    mod = np.sin(np.random.uniform(0.1, 0.5)*t)
    x = np.sin(np.random.uniform(2,5)*t + 2*mod)
    return x + np.random.randn(n)*noise

def gen_chirp(n, noise=0.05):
    """Fréquence qui augmente linéairement"""
    t = np.linspace(0, 1, n)
    f0, f1 = np.random.uniform(1,3), np.random.uniform(5,15)
    x = np.sin(2*np.pi*(f0*t + 0.5*(f1-f0)*t**2))
    return x + np.random.randn(n)*noise

def gen_damped_oscillator(n, noise=0.05):
    dt = 0.05
    gamma = np.random.uniform(0.01, 0.15)
    omega = np.random.uniform(0.5, 2.0)
    x, v = np.zeros(n), np.zeros(n)
    x[0] = np.random.uniform(-2, 2)
    v[0] = np.random.uniform(-1, 1)
    for t in range(1, n):
        a = -2*gamma*v[t-1] - omega**2*x[t-1]
        v[t] = v[t-1] + a*dt
        x[t] = x[t-1] + v[t]*dt + np.random.randn()*noise
    return x

def gen_growing_oscillator(n, noise=0.05):
    """Oscillateur instable — amplitude croissante"""
    t = np.linspace(0, 4*np.pi, n)
    growth = np.linspace(0.1, 1.5, n)
    f = np.random.uniform(1, 3)
    return growth * np.sin(f*t) + np.random.randn(n)*noise

def gen_coupled_oscillators(n, noise=0.05):
    """Deux oscillateurs couplés"""
    dt, k = 0.05, np.random.uniform(0.1, 0.5)
    x1, x2 = np.zeros(n), np.zeros(n)
    v1, v2 = np.zeros(n), np.zeros(n)
    x1[0], x2[0] = np.random.uniform(-1,1), np.random.uniform(-1,1)
    w1, w2 = np.random.uniform(0.5,2), np.random.uniform(0.5,2)
    for t in range(1, n):
        a1 = -w1**2*x1[t-1] + k*(x2[t-1]-x1[t-1])
        a2 = -w2**2*x2[t-1] + k*(x1[t-1]-x2[t-1])
        v1[t] = v1[t-1] + a1*dt
        x1[t] = x1[t-1] + v1[t]*dt
        v2[t] = v2[t-1] + a2*dt
        x2[t] = x2[t-1] + v2[t]*dt
    return x1 + np.random.randn(n)*noise

def gen_beating(n, noise=0.05):
    """Phénomène de battement — deux fréquences proches"""
    t = np.linspace(0, 8*np.pi, n)
    f1 = np.random.uniform(2, 4)
    f2 = f1 + np.random.uniform(0.05, 0.3)
    return np.sin(f1*t) + np.sin(f2*t) + np.random.randn(n)*noise

def gen_harmonic_series(n, noise=0.05):
    """Fondamentale + harmoniques — signal musical"""
    t = np.linspace(0, 4*np.pi, n)
    f0 = np.random.uniform(0.5, 1.5)
    x = sum(np.sin(k*f0*t) / k for k in range(1, 6))
    return x + np.random.randn(n)*noise

# ══════════════════════════════════════════════════════
# CATÉGORIE 2 — TENDANCES (10)
# ══════════════════════════════════════════════════════

def gen_linear_trend(n, noise=0.05):
    slope = np.random.uniform(-0.05, 0.05)
    return slope * np.arange(n) + np.random.randn(n)*noise

def gen_quadratic_trend(n, noise=0.05):
    a = np.random.uniform(-1e-4, 1e-4)
    t = np.arange(n)
    return a * t**2 + np.random.randn(n)*noise

def gen_exp_growth(n, noise=0.05):
    r = np.random.uniform(0.005, 0.02)
    x = np.exp(r * np.arange(n))
    return (x - x.mean()) / x.std() + np.random.randn(n)*noise

def gen_exp_decay(n, noise=0.05):
    r = np.random.uniform(0.005, 0.02)
    x = np.exp(-r * np.arange(n))
    return (x - x.mean()) / x.std() + np.random.randn(n)*noise

def gen_logistic_growth(n, noise=0.05):
    """Courbe en S — croissance vers une capacité max"""
    L = np.random.uniform(1.5, 3.0)
    k = np.random.uniform(0.05, 0.15)
    t0 = n // 2
    t = np.arange(n)
    x = L / (1 + np.exp(-k*(t-t0)))
    return x + np.random.randn(n)*noise

def gen_log_growth(n, noise=0.05):
    t = np.arange(1, n+1)
    a = np.random.uniform(0.5, 2.0)
    return a * np.log(t) + np.random.randn(n)*noise

def gen_power_law(n, noise=0.05):
    t = np.arange(1, n+1)
    a = np.random.uniform(0.3, 0.8)
    x = t**a
    return (x - x.mean()) / x.std() + np.random.randn(n)*noise

def gen_trend_seasonal(n, noise=0.05):
    """Trend linéaire + saisonnalité — le classique STL"""
    t = np.arange(n)
    trend = np.random.uniform(-0.02, 0.02) * t
    period = np.random.choice([12, 24, 52])
    seasonal = np.sin(2*np.pi*t/period)
    return trend + seasonal + np.random.randn(n)*noise

def gen_multiplicative_seasonal(n, noise=0.05):
    """Saisonnalité qui amplifie avec la tendance"""
    t = np.arange(n)
    trend = 1 + 0.01 * t
    period = np.random.choice([12, 24])
    seasonal = 1 + 0.3*np.sin(2*np.pi*t/period)
    x = trend * seasonal
    return (x - x.mean()) / x.std() + np.random.randn(n)*noise

def gen_double_seasonal(n, noise=0.05):
    """Deux saisonnalités imbriquées — électricité, trafic"""
    t = np.arange(n)
    s1 = np.sin(2*np.pi*t/24)       # journalière
    s2 = np.sin(2*np.pi*t/(24*7))   # hebdo
    return s1 + 0.5*s2 + np.random.randn(n)*noise

# ══════════════════════════════════════════════════════
# CATÉGORIE 3 — PROCESSUS STOCHASTIQUES (18)
# ══════════════════════════════════════════════════════

def gen_white_noise(n, noise=1.0):
    return np.random.randn(n) * noise

def gen_ar1(n, noise=0.1):
    phi = np.random.uniform(-0.95, 0.95)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi*x[t-1] + np.random.randn()*noise
    return x

def gen_ar2(n, noise=0.1):
    x = np.zeros(n)
    p1, p2 = np.random.uniform(-0.5, 0.5), np.random.uniform(-0.3, 0.3)
    for t in range(2, n):
        x[t] = p1*x[t-1] + p2*x[t-2] + np.random.randn()*noise
    return x

def gen_ma2(n, noise=0.1):
    eps = np.random.randn(n) * noise
    t1, t2 = np.random.uniform(-0.8, 0.8), np.random.uniform(-0.5, 0.5)
    x = np.zeros(n)
    for t in range(2, n):
        x[t] = eps[t] + t1*eps[t-1] + t2*eps[t-2]
    return x

def gen_arma(n, noise=0.1):
    x = np.zeros(n)
    eps = np.random.randn(n)*noise
    phi = np.random.uniform(-0.6, 0.6)
    theta = np.random.uniform(-0.6, 0.6)
    for t in range(1, n):
        x[t] = phi*x[t-1] + eps[t] + theta*eps[t-1]
    return x

def gen_random_walk(n, noise=0.1):
    return np.cumsum(np.random.randn(n) * noise)

def gen_random_walk_drift(n, noise=0.1):
    drift = np.random.uniform(-0.02, 0.02)
    return np.cumsum(drift + np.random.randn(n) * noise)

def gen_geometric_random_walk(n, noise=0.02):
    """GBM — modèle Black-Scholes pour les prix"""
    mu = np.random.uniform(-0.001, 0.001)
    returns = mu + np.random.randn(n)*noise
    x = np.exp(np.cumsum(returns))
    return (x - x.mean()) / x.std()

def gen_ornstein_uhlenbeck(n, noise=0.1):
    """Mean-reverting — taux d'intérêt, spreads"""
    theta = np.random.uniform(0.05, 0.3)
    mu = np.random.uniform(-0.5, 0.5)
    x = np.zeros(n)
    x[0] = mu + np.random.randn()
    for t in range(1, n):
        x[t] = x[t-1] + theta*(mu - x[t-1]) + np.random.randn()*noise
    return x

def gen_garch_like(n, noise=0.05):
    """Volatilité conditionnelle — clustering de vol financière"""
    alpha, beta = 0.1, 0.85
    sigma2 = np.ones(n) * 0.01
    x = np.zeros(n)
    for t in range(1, n):
        sigma2[t] = 0.001 + alpha*x[t-1]**2 + beta*sigma2[t-1]
        x[t] = np.random.randn() * np.sqrt(sigma2[t])
    return x

def gen_pink_noise(n, noise=0.05):
    """1/f noise — omniprésent dans la nature"""
    f = np.fft.rfftfreq(n)
    f[0] = 1e-10
    spectrum = np.random.randn(len(f)) / np.sqrt(f)
    x = np.fft.irfft(spectrum, n=n)
    return x / (x.std() + 1e-8) + np.random.randn(n)*noise

def gen_brown_noise(n, noise=0.05):
    """1/f² — intégration de bruit blanc"""
    x = np.cumsum(np.random.randn(n))
    return x / (x.std() + 1e-8) + np.random.randn(n)*noise

def gen_poisson_process(n, noise=0.02):
    """Arrivées discrètes — clicks, transactions"""
    lam = np.random.uniform(0.05, 0.3)
    x = np.random.poisson(lam, n).astype(float)
    return x + np.random.randn(n)*noise

def gen_hawkes_like(n, noise=0.05):
    """Self-exciting — les events clustérisent"""
    x = np.zeros(n)
    intensity = np.random.uniform(0.05, 0.15)
    decay = np.random.uniform(0.05, 0.2)
    for t in range(1, n):
        intensity = 0.05 + decay * x[t-1] + (1-decay)*intensity
        x[t] = float(np.random.poisson(intensity)) + np.random.randn()*noise
    return x

def gen_fractional_brownian(n, H_hurst=None, noise=0.02):
    """Longue mémoire — H proche de 1 = tendance persistante"""
    H_hurst = H_hurst or np.random.uniform(0.6, 0.9)
    # Approximation via filtrage AR long
    order = min(50, n//4)
    coeffs = np.array([(k**H_hurst - (k-1)**H_hurst) for k in range(1, order+1)])
    coeffs /= coeffs.sum()
    wn = np.random.randn(n)
    x = np.convolve(wn, coeffs, mode='full')[:n]
    return x / (x.std() + 1e-8) + np.random.randn(n)*noise

def gen_levy_flight(n, alpha=1.5, noise=0.02):
    """Sauts lourds — fat tails"""
    # Version numériquement robuste (évite puissance fractionnaire sur base négative / cos ~ 0)
    u = np.random.uniform(-np.pi/2, np.pi/2, n)
    w = np.random.exponential(1.0, n)
    w = np.maximum(w, 1e-12)
    cos_u = np.cos(u)
    cos_u = np.clip(cos_u, 1e-12, None)
    cos_term = np.cos(u*(1 - alpha))
    pow_term = (np.abs(cos_term) / w) ** ((1 - alpha) / alpha)
    steps = (np.sin(alpha*u) / (cos_u ** (1/alpha))) * np.sign(cos_term) * pow_term
    steps = np.clip(steps, -10, 10)
    x = np.cumsum(steps)
    return (x - x.mean()) / (x.std() + 1e-8) + np.random.randn(n)*noise

def gen_mixture_gaussian(n, noise=0.05):
    """Mélange de gaussiennes — bimodalité temporelle"""
    n_switches = np.random.randint(3, 10)
    switch_pts = np.sort(np.random.choice(n, n_switches, replace=False))
    x = np.zeros(n)
    mus = np.random.uniform(-2, 2, n_switches+1)
    stds = np.random.uniform(0.2, 0.8, n_switches+1)
    prev = 0
    for i, pt in enumerate(np.append(switch_pts, n)):
        x[prev:pt] = np.random.randn(pt-prev)*stds[i] + mus[i]
        prev = pt
    return x + np.random.randn(n)*noise

def gen_student_t_noise(n, noise=0.05):
    """Fat tails — queues épaisses"""
    df = np.random.uniform(2.5, 5.0)
    from numpy.random import standard_t
    return standard_t(df, size=n) * noise

# ══════════════════════════════════════════════════════
# CATÉGORIE 4 — SYSTÈMES CHAOTIQUES (10)
# ══════════════════════════════════════════════════════

def gen_lorenz_x(n, noise=0.02):
    dt, s, r, b = 0.01, 10, 28, 8/3
    x, y, z = np.random.uniform(-10,10), np.random.uniform(-10,10), np.random.uniform(20,30)
    out = np.zeros(n)
    for t in range(n):
        dx, dy, dz = s*(y-x), x*(r-z)-y, x*y-b*z
        x+=dx*dt
        y+=dy*dt
        z+=dz*dt
        out[t] = x
    out = (out-out.mean())/(out.std()+1e-8)
    return out + np.random.randn(n)*noise

def gen_lorenz_z(n, noise=0.02):
    dt, s, r, b = 0.01, 10, 28, 8/3
    x, y, z = np.random.uniform(-10,10), np.random.uniform(-10,10), np.random.uniform(20,30)
    out = np.zeros(n)
    for t in range(n):
        dx, dy, dz = s*(y-x), x*(r-z)-y, x*y-b*z
        x+=dx*dt
        y+=dy*dt
        z+=dz*dt
        out[t] = z
    out = (out-out.mean())/(out.std()+1e-8)
    return out + np.random.randn(n)*noise

def gen_rossler(n, noise=0.02):
    dt, a, b, c = 0.02, 0.2, 0.2, 5.7
    x, y, z = np.random.randn(), np.random.randn(), np.random.uniform(0,5)
    out = np.zeros(n)
    for t in range(n):
        dx, dy, dz = -y-z, x+a*y, b+z*(x-c)
        x+=dx*dt
        y+=dy*dt
        z+=dz*dt
        out[t] = x
    return (out-out.mean())/(out.std()+1e-8) + np.random.randn(n)*noise

def gen_van_der_pol(n, noise=0.02):
    dt, mu = 0.05, np.random.uniform(0.5, 3.0)
    x, v = np.random.randn(), np.random.randn()
    out = np.zeros(n)
    for t in range(n):
        dv = mu*(1-x**2)*v - x
        x += v*dt
        v += dv*dt
        out[t] = x
    return (out-out.mean())/(out.std()+1e-8) + np.random.randn(n)*noise

def gen_duffing(n, noise=0.02):
    dt, alpha, beta, delta, gamma, omega = 0.05, 1, -1, 0.3, 0.5, 1.2
    x, v, t_run = np.random.randn()*0.5, 0.0, 0.0
    out = np.zeros(n)
    # Sécurité numérique: le terme x**3 peut diverger et overflow
    x_max = 50.0
    v_max = 200.0
    for i in range(n):
        f = gamma*np.cos(omega*t_run)
        x_use = float(np.clip(x, -x_max, x_max))
        v_use = float(np.clip(v, -v_max, v_max))
        dv = f - delta*v_use - alpha*x_use - beta*(x_use**3)
        x += v_use*dt
        v += dv*dt
        t_run += dt
        x = float(np.clip(x, -x_max, x_max))
        v = float(np.clip(v, -v_max, v_max))
        out[i] = x
    return (out-out.mean())/(out.std()+1e-8) + np.random.randn(n)*noise

def gen_logistic_map(n, noise=0.02):
    """Chaos déterministe 1D"""
    r = np.random.uniform(3.7, 4.0)
    x = np.zeros(n)
    x[0] = np.random.uniform(0.1, 0.9)
    for t in range(1, n):
        x[t] = r * x[t-1] * (1 - x[t-1])
    return (x - 0.5) / 0.3 + np.random.randn(n)*noise

def gen_henon_map(n, noise=0.02):
    a, b = 1.4, 0.3
    x, y = np.random.randn()*0.1, np.random.randn()*0.1
    out = np.zeros(n)
    for t in range(n):
        xn = 1 - a*x**2 + y
        y = b*x
        x = xn
        out[t] = x
    return (out-out.mean())/(out.std()+1e-8) + np.random.randn(n)*noise

def gen_mackey_glass(n, noise=0.02, tau=17):
    """Équation différentielle à délai — utilisée en benchmark TS"""
    beta, gamma, n_mg = 0.2, 0.1, 10
    x = np.ones(n + tau) * 1.2
    for t in range(tau, n + tau):
        x[t] = x[t-1] + (beta*x[t-tau])/(1+x[t-tau]**n_mg) - gamma*x[t-1]
    out = x[tau:]
    return (out-out.mean())/(out.std()+1e-8) + np.random.randn(n)*noise

def gen_tent_map(n, noise=0.02):
    mu = np.random.uniform(1.5, 2.0)
    x = np.zeros(n)
    x[0] = np.random.uniform(0.1, 0.9)
    for t in range(1, n):
        x[t] = mu*x[t-1] if x[t-1] < 0.5 else mu*(1-x[t-1])
    return (x - x.mean())/(x.std()+1e-8) + np.random.randn(n)*noise

def gen_ikeda_map(n, noise=0.02):
    u = np.random.uniform(0.7, 0.9)
    x, y = np.random.randn()*0.1, np.random.randn()*0.1
    out = np.zeros(n)
    for t in range(n):
        t_n = 0.4 - 6/(1+x**2+y**2)
        xn = 1 + u*(x*np.cos(t_n) - y*np.sin(t_n))
        y = u*(x*np.sin(t_n) + y*np.cos(t_n))
        x = xn
        out[t] = x
    return (out-out.mean())/(out.std()+1e-8) + np.random.randn(n)*noise

# ══════════════════════════════════════════════════════
# CATÉGORIE 5 — SYSTÈMES PHYSIQUES & ODEs (12)
# ══════════════════════════════════════════════════════

def gen_lotka_volterra(n, noise=0.05):
    """Proie-prédateur — écologie"""
    dt = 0.05
    alpha = np.random.uniform(0.8, 1.2)
    beta  = np.random.uniform(0.1, 0.3)
    delta = np.random.uniform(0.1, 0.3)
    gamma = np.random.uniform(0.8, 1.2)
    x, y = np.random.uniform(1, 3), np.random.uniform(1, 3)
    out = np.zeros(n)
    for t in range(n):
        dx = (alpha - beta*y)*x
        dy = (delta*x - gamma)*y
        x += dx*dt
        y += dy*dt
        x = max(x, 0.01)
        y = max(y, 0.01)
        out[t] = x
    return (out-out.mean())/(out.std()+1e-8) + np.random.randn(n)*noise

def gen_sir_epidemic(n, noise=0.02):
    """Modèle épidémique SIR — retourne I(t)"""
    dt = 0.1
    beta_ep = np.random.uniform(0.2, 0.5)
    gamma_ep = np.random.uniform(0.05, 0.15)
    S, Id, R = 0.99, 0.01, 0.0
    out = np.zeros(n)
    for t in range(n):
        dS = -beta_ep*S*Id
        dI = beta_ep*S*Id - gamma_ep*Id
        dR = gamma_ep*Id
        S+=dS*dt
        Id+=dI*dt
        R+=dR*dt
        Id = max(Id, 0)
        out[t] = Id
    out = (out-out.mean())/(out.std()+1e-8)
    return out + np.random.randn(n)*noise

def gen_rc_circuit(n, noise=0.02):
    """Réponse impulsionnelle RC"""
    dt, RC = 0.01, np.random.uniform(0.1, 0.5)
    input_signal = np.random.randn(n)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = x[t-1] + dt/RC*(input_signal[t] - x[t-1])
    return x + np.random.randn(n)*noise

def gen_spring_mass(n, noise=0.05):
    """Ressort-masse-amortisseur"""
    dt = 0.05
    k = np.random.uniform(0.5, 3.0)
    m = np.random.uniform(0.5, 2.0)
    c = np.random.uniform(0.05, 0.5)
    x, v = np.random.uniform(-2, 2), 0.0
    out = np.zeros(n)
    for t in range(n):
        a = (-k*x - c*v) / m
        v += a*dt
        x += v*dt
        out[t] = x
    return (out-out.mean())/(out.std()+1e-8) + np.random.randn(n)*noise

def gen_pendulum(n, noise=0.02):
    dt, g, L = 0.05, 9.81, np.random.uniform(0.5, 2.0)
    theta = np.random.uniform(-np.pi/3, np.pi/3)
    omega = 0.0
    out = np.zeros(n)
    for t in range(n):
        alpha = -(g/L)*np.sin(theta)
        omega += alpha*dt
        theta += omega*dt
        out[t] = theta
    return (out-out.mean())/(out.std()+1e-8) + np.random.randn(n)*noise

def gen_forced_oscillator(n, noise=0.05):
    """Oscillateur forcé — résonance possible"""
    dt = 0.05
    omega0 = np.random.uniform(0.5, 2.0)
    omega_f = np.random.uniform(0.3, 2.5)
    gamma = np.random.uniform(0.01, 0.2)
    F = np.random.uniform(0.5, 2.0)
    x, v, t_r = 0.0, 0.0, 0.0
    out = np.zeros(n)
    for t in range(n):
        a = F*np.cos(omega_f*t_r) - 2*gamma*v - omega0**2*x
        v += a*dt
        x += v*dt
        t_r += dt
        out[t] = x
    return (out-out.mean())/(out.std()+1e-8) + np.random.randn(n)*noise

def gen_heat_equation_1d(n, noise=0.02):
    """Diffusion thermique 1D — on observe un point"""
    alpha, dx, dt = 0.01, 0.1, 0.001
    size = 50
    u = np.zeros(size)
    u[size//2] = 10.0
    out = np.zeros(n)
    r = alpha*dt/dx**2
    for t in range(n):
        u_new = u.copy()
        u_new[1:-1] = u[1:-1] + r*(u[2:] - 2*u[1:-1] + u[:-2])
        u = u_new
        out[t] = u[size//4]
    return (out-out.mean())/(out.std()+1e-8) + np.random.randn(n)*noise

def gen_cir_process(n, noise=0.02):
    """Cox-Ingersoll-Ross — taux d'intérêt positifs"""
    dt = 0.01
    kappa = np.random.uniform(0.5, 2.0)
    theta = np.random.uniform(0.02, 0.08)
    sigma = np.random.uniform(0.01, 0.05)
    x = theta
    out = np.zeros(n)
    for t in range(n):
        x += kappa*(theta-x)*dt + sigma*np.sqrt(max(x,0))*np.random.randn()*np.sqrt(dt)
        x = max(x, 0)
        out[t] = x
    return (out-out.mean())/(out.std()+1e-8)

def gen_glycolysis(n, noise=0.02):
    """Oscillateur biochimique de Sel'kov — glycolyse"""
    dt, a, b = 0.05, np.random.uniform(0.05,0.1), np.random.uniform(0.5,0.8)
    x, y = np.random.uniform(0.5,1.5), np.random.uniform(0.5,1.5)
    out = np.zeros(n)
    for t in range(n):
        dx = -x + a*y + x**2*y
        dy = b - a*y - x**2*y
        x += dx*dt
        y += dy*dt
        out[t] = x
    return (out-out.mean())/(out.std()+1e-8) + np.random.randn(n)*noise

def gen_brusselator(n, noise=0.02):
    """Réaction chimique oscillante"""
    dt, A, B = 0.05, np.random.uniform(1,2), np.random.uniform(2,4)
    x, y = 1.0, 1.0
    out = np.zeros(n)
    for t in range(n):
        dx = A - (B+1)*x + x**2*y
        dy = B*x - x**2*y
        x += dx*dt
        y += dy*dt
        out[t] = x
    return (out-out.mean())/(out.std()+1e-8) + np.random.randn(n)*noise

def gen_fitzhugh_nagumo(n, noise=0.02):
    """Modèle de neurone"""
    dt = 0.1
    a, b, tau, Id = 0.7, 0.8, 12.5, np.random.uniform(0.5, 1.0)
    v, w = np.random.randn()*0.1, np.random.randn()*0.1
    out = np.zeros(n)
    for t in range(n):
        dv = v - v**3/3 - w + Id
        dw = (v + a - b*w)/tau
        v += dv*dt
        w += dw*dt
        out[t] = v
    return (out-out.mean())/(out.std()+1e-8) + np.random.randn(n)*noise

def gen_double_well(n, noise=0.1):
    """Potentiel double puits — transitions entre 2 états"""
    dt = 0.01
    x = np.random.choice([-1.0, 1.0])
    out = np.zeros(n)
    for t in range(n):
        force = -(-2*x + 4*x**3)  # gradient de -(x²-x⁴)
        x += force*dt + np.random.randn()*noise*np.sqrt(dt)
        out[t] = x
    return out

# ══════════════════════════════════════════════════════
# CATÉGORIE 6 — RÉGIMES & TRANSITIONS (12)
# ══════════════════════════════════════════════════════

def gen_regime_switch_2(n, noise=0.05):
    """Bascule entre 2 dynamiques"""
    k = np.random.randint(n//3, 2*n//3)
    gens = list(GENERATORS.values())
    x1 = np.random.choice(gens)(k, noise=noise)
    x2 = np.random.choice(gens)(n-k, noise=noise)
    return np.concatenate([x1, x2])

def gen_regime_switch_3(n, noise=0.05):
    """3 régimes distincts"""
    k1, k2 = sorted(np.random.choice(range(n//4, 3*n//4), 2, replace=False))
    gens = list(GENERATORS.values())
    def g(m: int):
        return np.random.choice(gens)(m, noise=noise)
    return np.concatenate([g(k1), g(k2 - k1), g(n - k2)])

def gen_level_shift(n, noise=0.05):
    """Saut de niveau soudain"""
    k = np.random.randint(n//3, 2*n//3)
    shift = np.random.uniform(1.0, 3.0) * np.random.choice([-1, 1])
    x = np.random.randn(n)*noise
    x[k:] += shift
    return x

def gen_variance_shift(n, noise=0.05):
    """La variabilité change — CUSUM target"""
    k = np.random.randint(n//3, 2*n//3)
    x = np.zeros(n)
    x[:k] = np.random.randn(k)*0.1
    x[k:] = np.random.randn(n-k)*1.0
    return x

def gen_trend_change(n, noise=0.05):
    """Changement de pente"""
    k = np.random.randint(n//3, 2*n//3)
    s1, s2 = np.random.uniform(-0.05, 0.05), np.random.uniform(-0.05, 0.05)
    t = np.arange(n)
    x = np.where(t < k, s1*t, s1*k + s2*(t-k))
    return x + np.random.randn(n)*noise

def gen_intermittent(n, noise=0.05):
    """Demande intermittente — retail"""
    sparsity = np.random.uniform(0.5, 0.9)
    x = np.random.exponential(1.0, n)
    x[np.random.rand(n) < sparsity] = 0.0
    return x + np.random.randn(n)*noise*0.1

def gen_spike_series(n, noise=0.05):
    """Séries avec pics rares — événements"""
    x = np.random.randn(n)*noise
    n_spikes = np.random.randint(3, 15)
    positions = np.random.choice(n, n_spikes, replace=False)
    amplitudes = np.random.uniform(2, 6, n_spikes)
    x[positions] += amplitudes
    return x

def gen_step_function(n, noise=0.05):
    """Escalier — changements discrets"""
    n_steps = np.random.randint(3, 10)
    breakpoints = np.sort(np.random.choice(n, n_steps, replace=False))
    levels = np.cumsum(np.random.randn(n_steps+1)*0.5)
    x = np.zeros(n)
    prev = 0
    for i, bp in enumerate(np.append(breakpoints, n)):
        x[prev:bp] = levels[i]
        prev = bp
    return x + np.random.randn(n)*noise

def gen_markov_switching(n, noise=0.1):
    """Modèle de Markov à 3 états"""
    means = np.random.uniform(-2, 2, 3)
    stds  = np.random.uniform(0.1, 0.8, 3)
    P = np.abs(np.random.randn(3, 3))
    P /= P.sum(axis=1, keepdims=True)
    state = 0
    x = np.zeros(n)
    for t in range(n):
        state = np.random.choice(3, p=P[state])
        x[t] = np.random.randn()*stds[state] + means[state]
    return x

def gen_seasonal_regime(n, noise=0.05):
    """Saisonnalité avec régime différent en hiver/été"""
    t = np.arange(n)
    period = np.random.choice([24, 52, 12])
    phase = (t % period) / period
    amp = 1 + 0.5*np.sin(2*np.pi*phase)
    freq = np.random.uniform(1, 3)
    x = amp * np.sin(2*np.pi*freq*phase)
    return x + np.random.randn(n)*noise

def gen_pulse_train(n, noise=0.05):
    """Train d'impulsions périodiques"""
    period = np.random.randint(10, 40)
    width  = np.random.randint(2, period//3)
    x = np.zeros(n)
    for i in range(0, n, period):
        x[i:i+width] = np.random.uniform(0.5, 2.0)
    return x + np.random.randn(n)*noise

def gen_ramp_signals(n, noise=0.05):
    """Rampes successives"""
    x = np.zeros(n)
    t = 0
    while t < n:
        length = np.random.randint(20, n//4)
        slope  = np.random.uniform(-0.1, 0.1)
        seg    = slope * np.arange(min(length, n-t))
        x[t:t+len(seg)] = x[t-1] + seg if t > 0 else seg
        t += length
    return x + np.random.randn(n)*noise

# ══════════════════════════════════════════════════════
# CATÉGORIE 7 — SÉRIES RÉELLES SIMULÉES (15)
# ══════════════════════════════════════════════════════

def gen_electricity_like(n, noise=0.05):
    """Consommation électrique — double saisonnalité + trend"""
    t = np.arange(n)
    daily    = np.sin(2*np.pi*t/24)
    weekly   = 0.5*np.sin(2*np.pi*t/168)
    trend    = 0.001*t
    peak     = 0.3*np.exp(-((t % 24 - 18)**2)/8)
    return trend + daily + weekly + peak + np.random.randn(n)*noise

def gen_temperature_like(n, noise=0.1):
    """Température — annuel + diurne + trend climat"""
    t = np.arange(n)
    annual  = 10*np.sin(2*np.pi*t/365)
    diurnal = 3*np.sin(2*np.pi*t/24 - np.pi/4)
    trend   = 0.0001*t
    return annual + diurnal + trend + np.random.randn(n)*noise

def gen_traffic_like(n, noise=0.1):
    """Trafic web/routier — rush hours + weekend effect"""
    t = np.arange(n)
    hour_of_day = t % 24
    rush = np.exp(-((hour_of_day-8)**2)/4) + np.exp(-((hour_of_day-18)**2)/4)
    weekend = (t % 168 >= 120).astype(float) * 0.4
    return rush - weekend + np.random.randn(n)*noise

def gen_financial_returns(n, noise=0.01):
    """Rendements financiers — fat tails + vol clustering"""
    return gen_garch_like(n, noise) * np.random.uniform(0.5, 2.0)

def gen_stock_price(n, noise=0.02):
    """Prix d'action — GBM avec trend"""
    return gen_geometric_random_walk(n, noise)

def gen_retail_sales(n, noise=0.1):
    """Ventes retail — trend + saisonnalité + promotions"""
    t = np.arange(n)
    trend    = 0.005*t
    weekly   = np.sin(2*np.pi*t/7)
    promo    = np.zeros(n)
    for i in np.random.choice(n, n//20, replace=False):
        promo[i:min(i+3,n)] = np.random.uniform(0.5, 2.0)
    return trend + weekly + promo + np.random.randn(n)*noise

def gen_eeg_like(n, noise=0.05):
    """Signal EEG simulé — oscillations cérébrales"""
    t = np.linspace(0, 10, n)
    alpha = 2*np.sin(2*np.pi*10*t)
    beta  = 0.5*np.sin(2*np.pi*20*t)
    theta = 1.0*np.sin(2*np.pi*5*t)
    return alpha + beta + theta + np.random.randn(n)*noise*2

def gen_ecg_like(n, noise=0.02):
    """ECG simplifié — complexe QRS périodique"""
    hr = np.random.uniform(60, 100)
    period = int(60/hr * 100)
    t_cycle = np.linspace(0, 2*np.pi, period)
    qrs = (np.exp(-((t_cycle-1.5)**2)/0.1) -
           0.3*np.exp(-((t_cycle-1.2)**2)/0.3) -
           0.3*np.exp(-((t_cycle-1.8)**2)/0.3))
    x = np.tile(qrs, n//period + 1)[:n]
    return x + np.random.randn(n)*noise

def gen_wind_speed(n, noise=0.1):
    """Vitesse du vent — Weibull + trend diurne"""
    k = np.random.uniform(1.5, 2.5)
    scale = np.random.uniform(5, 15)
    x = np.random.weibull(k, n) * scale
    t = np.arange(n)
    diurnal = 2*np.sin(2*np.pi*t/24 - np.pi/3)
    x += diurnal
    return (x - x.mean())/(x.std()+1e-8) + np.random.randn(n)*noise

def gen_precipitation(n, noise=0.05):
    """Précipitations — Gamma + saisonnalité"""
    seasonal = 1 + 0.5*np.sin(2*np.pi*np.arange(n)/365)
    x = np.random.gamma(0.5, 1, n) * seasonal
    x[np.random.rand(n) < 0.6] = 0.0
    return (x - x.mean())/(x.std()+1e-8) + np.random.randn(n)*noise

def gen_cpu_usage(n, noise=0.05):
    """Usage CPU — baseline + spikes + périodicité"""
    t = np.arange(n)
    base = 0.3 + 0.1*np.sin(2*np.pi*t/24)
    x = base + np.random.randn(n)*0.05
    n_jobs = np.random.randint(3, 10)
    for _ in range(n_jobs):
        start = np.random.randint(0, n)
        dur   = np.random.randint(5, 50)
        load  = np.random.uniform(0.3, 0.6)
        x[start:start+dur] = np.clip(x[start:start+dur] + load, 0, 1)
    return x + np.random.randn(n)*noise

def gen_network_traffic(n, noise=0.05):
    """Trafic réseau — burst + longue mémoire"""
    base = gen_pink_noise(n, noise=0.02)
    bursts = np.zeros(n)
    for _ in range(np.random.randint(5, 20)):
        pos = np.random.randint(0, n)
        dur = np.random.randint(3, 20)
        bursts[pos:pos+dur] += np.random.exponential(2.0)
    return base + bursts

def gen_sensor_drift(n, noise=0.05):
    """Capteur qui dérive — IoT, maintenance prédictive"""
    slow_drift = 0.003 * np.arange(n) * np.random.choice([-1, 1])
    periodic   = 0.2*np.sin(2*np.pi*np.arange(n)/100)
    x = slow_drift + periodic + np.random.randn(n)*noise
    if np.random.rand() > 0.5:
        k = np.random.randint(n//2, 3*n//4)
        x[k:] += np.random.uniform(0.5, 1.5)
    return x

def gen_exchange_rate(n, noise=0.005):
    """Taux de change — proche random walk, léger mean-revert"""
    return gen_ornstein_uhlenbeck(n, noise=noise)

def gen_pageviews(n, noise=0.1):
    """Pageviews — trend + double saisonnalité + virality"""
    t = np.arange(n)
    trend = np.log(1 + 0.002*t)
    daily = np.sin(2*np.pi*t/24)
    weekly = 0.4*np.sin(2*np.pi*t/168)
    x = trend + daily + weekly
    # Virality spike
    if np.random.rand() > 0.5:
        k = np.random.randint(n//4, 3*n//4)
        x[k:k+10] += np.random.uniform(2, 5)
    return x + np.random.randn(n)*noise

# ══════════════════════════════════════════════════════
# CATÉGORIE 8 — COMPOSITIONS & MIXTURES (8)
# ══════════════════════════════════════════════════════

def gen_sine_ar_residuals(n, noise=0.1):
    """Signal propre + résidus AR"""
    signal = gen_sine(n, noise=0.0)
    resids = gen_ar2(n, noise=noise)
    return signal + 0.3*resids

def gen_trend_garch(n, noise=0.02):
    """Trend avec volatilité conditionnelle"""
    trend = gen_linear_trend(n, noise=0.0)
    vol   = gen_garch_like(n, noise=noise)
    return trend + vol

def gen_seasonal_regime_switch(n, noise=0.05):
    """Saisonnalité + bascule de régime"""
    seasonal = gen_trend_seasonal(n, noise=0.02)
    k = np.random.randint(n//3, 2*n//3)
    seasonal[k:] += np.random.uniform(0.5, 1.5)
    return seasonal + np.random.randn(n)*noise

def gen_lorenz_periodic_forcing(n, noise=0.02):
    """Chaos + forçage périodique"""
    chaos  = gen_lorenz_x(n, noise=0.0)
    period = gen_sine(n, noise=0.0)
    return 0.7*chaos + 0.3*period + np.random.randn(n)*noise

def gen_multi_scale(n, noise=0.05):
    """Plusieurs échelles temporelles"""
    t = np.arange(n)
    slow   = np.sin(2*np.pi*t / (n//3))
    medium = 0.5*np.sin(2*np.pi*t / 50)
    fast   = 0.2*np.sin(2*np.pi*t / 10)
    return slow + medium + fast + np.random.randn(n)*noise

def gen_rw_mean_revert(n, noise=0.1):
    """Random walk + mean reversion — régimes mixtes"""
    x = np.zeros(n)
    mu = 0.0
    for t in range(1, n):
        rw_component = np.random.randn()*noise
        mr_component = -0.1*(x[t-1] - mu)
        x[t] = x[t-1] + rw_component + mr_component
    return x

def gen_noisy_ode(n, noise=0.2):
    """ODE propre mais très bruitée — challenge de débruitage"""
    return gen_damped_oscillator(n, noise=noise)

def gen_mixed_frequency(n, noise=0.05):
    """Signal mixant basse et haute fréquence"""
    t = np.linspace(0, 10, n)
    low  = np.sin(0.5*t)
    high = 0.3*np.sin(15*t)
    return low + high + np.random.randn(n)*noise

# ══════════════════════════════════════════════════════
# REGISTRE GLOBAL — 100 générateurs
# ══════════════════════════════════════════════════════

GENERATORS = {
    # Oscillateurs (15)
    "sine":                 gen_sine,
    "cosine":               gen_cosine,
    "square_wave":          gen_square_wave,
    "sawtooth":             gen_sawtooth,
    "triangle_wave":        gen_triangle_wave,
    "multi_sine":           gen_multi_sine,
    "am_signal":            gen_am_signal,
    "fm_signal":            gen_fm_signal,
    "chirp":                gen_chirp,
    "damped_oscillator":    gen_damped_oscillator,
    "growing_oscillator":   gen_growing_oscillator,
    "coupled_oscillators":  gen_coupled_oscillators,
    "beating":              gen_beating,
    "harmonic_series":      gen_harmonic_series,
    # Tendances (10)
    "linear_trend":         gen_linear_trend,
    "quadratic_trend":      gen_quadratic_trend,
    "exp_growth":           gen_exp_growth,
    "exp_decay":            gen_exp_decay,
    "logistic_growth":      gen_logistic_growth,
    "log_growth":           gen_log_growth,
    "power_law":            gen_power_law,
    "trend_seasonal":       gen_trend_seasonal,
    "multiplicative_seas":  gen_multiplicative_seasonal,
    "double_seasonal":      gen_double_seasonal,
    # Stochastiques (18)
    "white_noise":          gen_white_noise,
    "ar1":                  gen_ar1,
    "ar2":                  gen_ar2,
    "ma2":                  gen_ma2,
    "arma":                 gen_arma,
    "random_walk":          gen_random_walk,
    "rw_drift":             gen_random_walk_drift,
    "gbm":                  gen_geometric_random_walk,
    "ornstein_uhlenbeck":   gen_ornstein_uhlenbeck,
    "garch":                gen_garch_like,
    "pink_noise":           gen_pink_noise,
    "brown_noise":          gen_brown_noise,
    "poisson":              gen_poisson_process,
    "hawkes":               gen_hawkes_like,
    "fractional_bm":        gen_fractional_brownian,
    "levy_flight":          gen_levy_flight,
    "mixture_gaussian":     gen_mixture_gaussian,
    "student_t":            gen_student_t_noise,
    # Chaotiques (10)
    "lorenz_x":             gen_lorenz_x,
    "lorenz_z":             gen_lorenz_z,
    "rossler":              gen_rossler,
    "van_der_pol":          gen_van_der_pol,
    "duffing":              gen_duffing,
    "logistic_map":         gen_logistic_map,
    "henon_map":            gen_henon_map,
    "mackey_glass":         gen_mackey_glass,
    "tent_map":             gen_tent_map,
    "ikeda_map":            gen_ikeda_map,
    # Physiques / ODEs (12)
    "lotka_volterra":       gen_lotka_volterra,
    "sir_epidemic":         gen_sir_epidemic,
    "rc_circuit":           gen_rc_circuit,
    "spring_mass":          gen_spring_mass,
    "pendulum":             gen_pendulum,
    "forced_oscillator":    gen_forced_oscillator,
    "heat_equation":        gen_heat_equation_1d,
    "cir_process":          gen_cir_process,
    "glycolysis":           gen_glycolysis,
    "brusselator":          gen_brusselator,
    "fitzhugh_nagumo":      gen_fitzhugh_nagumo,
    "double_well":          gen_double_well,
    # Régimes (12)
    "regime_switch_2":      gen_regime_switch_2,
    "regime_switch_3":      gen_regime_switch_3,
    "level_shift":          gen_level_shift,
    "variance_shift":       gen_variance_shift,
    "trend_change":         gen_trend_change,
    "intermittent":         gen_intermittent,
    "spike_series":         gen_spike_series,
    "step_function":        gen_step_function,
    "markov_switching":     gen_markov_switching,
    "seasonal_regime":      gen_seasonal_regime,
    "pulse_train":          gen_pulse_train,
    "ramp_signals":         gen_ramp_signals,
    # Réels simulés (15)
    "electricity":          gen_electricity_like,
    "temperature":          gen_temperature_like,
    "traffic":              gen_traffic_like,
    "financial_returns":    gen_financial_returns,
    "stock_price":          gen_stock_price,
    "retail_sales":         gen_retail_sales,
    "eeg":                  gen_eeg_like,
    "ecg":                  gen_ecg_like,
    "wind_speed":           gen_wind_speed,
    "precipitation":        gen_precipitation,
    "cpu_usage":            gen_cpu_usage,
    "network_traffic":      gen_network_traffic,
    "sensor_drift":         gen_sensor_drift,
    "exchange_rate":        gen_exchange_rate,
    "pageviews":            gen_pageviews,
    # Compositions (8)
    "sine_ar":              gen_sine_ar_residuals,
    "trend_garch":          gen_trend_garch,
    "seasonal_regime_sw":   gen_seasonal_regime_switch,
    "lorenz_periodic":      gen_lorenz_periodic_forcing,
    "multi_scale":          gen_multi_scale,
    "rw_mean_revert":       gen_rw_mean_revert,
    "noisy_ode":            gen_noisy_ode,
    "mixed_frequency":      gen_mixed_frequency,
}

print(f"Total : {len(GENERATORS)} générateurs")   # → 100


# ══════════════════════════════════════════════════════
# PIPELINE : normalisation + fenêtrage + dataset
# ══════════════════════════════════════════════════════

def normalize(x: np.ndarray) -> np.ndarray:
    p5, p95 = np.percentile(x, 5), np.percentile(x, 95)
    rng = p95 - p5 + 1e-8
    return np.clip((x - p5) / rng, -5, 5).astype(np.float32)

def make_windows(series, Q, H, stride=1):
    N = (len(series) - Q - H) // stride
    past   = np.stack([series[i*stride       : i*stride+Q]   for i in range(N)])
    future = np.stack([series[i*stride+Q     : i*stride+Q+H] for i in range(N)])
    return past, future

def build_dataloaders(
    X_past:   np.ndarray,
    X_future: np.ndarray,
    cfg,
    train_ratio: float = 0.7,
    val_ratio:   float = 0.15,
    # test = 1 - train - val = 0.15
    seed: int = 42,
):
    """
    Parameters
    ----------
    X_past   : (N, Q, C)
    X_future : (N, H, C)
    cfg      : Config  — needs cfg.batch_size

    Returns
    -------
    train_dl, val_dl, test_dl : DataLoaders
    split_indices             : dict with keys "train", "val", "test"
                                each containing the original row indices
    """
    N = len(X_past)
    assert len(X_future) == N

    # ── Convert to tensors ──────────────────────────────────────
    X = torch.tensor(X_past,   dtype=torch.float32)
    Y = torch.tensor(X_future, dtype=torch.float32)
    ds = TensorDataset(X, Y)

    # ── Reproducible shuffle ────────────────────────────────────
    rng  = np.random.default_rng(seed)
    perm = rng.permutation(N)

    n_train = int(N * train_ratio)
    n_val   = int(N * val_ratio)

    idx_train = perm[:n_train]
    idx_val   = perm[n_train : n_train + n_val]
    idx_test  = perm[n_train + n_val:]

    # ── DataLoaders ─────────────────────────────────────────────
    train_dl = DataLoader(
        Subset(ds, idx_train),
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_dl = DataLoader(
        Subset(ds, idx_val),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_dl = DataLoader(
        Subset(ds, idx_test),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
    )

    split_indices = {
        "train": idx_train,
        "val":   idx_val,
        "test":  idx_test,
    }

    return train_dl, val_dl, test_dl, split_indices

def build_pretrain_dataset(n_series=5000, series_len=500, Q=96, H=24, stride=10):
    all_past, all_future, all_labels = [], [], []
    kinds = list(GENERATORS.keys())
    for _ in range(n_series):
        kind = np.random.choice(kinds)
        try:
            raw = GENERATORS[kind](series_len)
            series = normalize(raw)
            if np.isnan(series).any() or np.isinf(series).any():
                continue
            xp, xf = make_windows(series, Q, H, stride)
            all_past.append(xp)
            all_future.append(xf)
            all_labels.extend([kind]*len(xp))
        except Exception:
            continue   # quelques ODEs instables, on skip

    X_past   = np.concatenate(all_past)
    X_future = np.concatenate(all_future)
    idx = np.random.permutation(len(X_past))
    print(f"Dataset : {len(X_past):,} fenêtres | {len(set(all_labels))} types")
    return X_past[idx], X_future[idx], [all_labels[i] for i in idx]
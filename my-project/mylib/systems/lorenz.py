import numpy as np
from typing import Callable, Optional, Union

FloatOrFn = Union[float, Callable[[float], float]]
ForceFn = Callable[[float, np.ndarray], np.ndarray]

def _val(v: FloatOrFn, t: float) -> float:
    return float(v(t)) if callable(v) else float(v)

def lorenz_rhs(s: np.ndarray, sigma: float, rho: float, beta: float) -> np.ndarray:
    x, y, z = s
    return np.array([sigma*(y-x), x*(rho-z)-y, x*y-beta*z], dtype=float)

def rk4_step(
    s: np.ndarray,
    t: float,
    dt: float,
    sigma: FloatOrFn,
    rho: FloatOrFn,
    beta: FloatOrFn,
    force: Optional[ForceFn] = None,
) -> np.ndarray:
    def f(tt: float, ss: np.ndarray) -> np.ndarray:
        sig = _val(sigma, tt)
        r = _val(rho, tt)
        b = _val(beta, tt)
        ds = lorenz_rhs(ss, sig, r, b)
        if force is not None:
            ds = ds + force(tt, ss)
        return ds

    k1 = f(t, s)
    k2 = f(t + 0.5*dt, s + 0.5*dt*k1)
    k3 = f(t + 0.5*dt, s + 0.5*dt*k2)
    k4 = f(t + dt, s + dt*k3)
    return s + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def simulate(
    s0=(0.0, 1.0, 1.05),
    duration: float = 5.0,
    dt: float = 0.01,
    sigma: FloatOrFn = 10.0,
    rho: FloatOrFn = 28.0,
    beta: FloatOrFn = 8.0/3.0,
    burn_in: float = 2.0,
    force: Optional[ForceFn] = None,
    max_norm: float = 1e3,
):
    """
    Run RK4 simulation and return (points[N,3], s_end).
    burn_in: seconds to discard at the beginning (stabilize on attractor).
    """
    s = np.array(s0, dtype=float)
    out = []
    t = 0.0
    burn_steps = int(burn_in / dt)
    steps = int(duration / dt)

    # burn-in
    for _ in range(burn_steps):
        s = rk4_step(s, t, dt, sigma, rho, beta, force=None)
        if np.linalg.norm(s) > max_norm:
            s = s / np.linalg.norm(s) * max_norm
        t += dt

    # keep
    for _ in range(steps):
        s = rk4_step(s, t, dt, sigma, rho, beta, force=force)
        if np.linalg.norm(s) > max_norm:
            s = s / np.linalg.norm(s) * max_norm
        out.append(s.copy())
        t += dt

    return np.array(out), s

def smoothstep(u: float) -> float:
    u = max(0.0, min(1.0, u))
    return u*u*(3 - 2*u)

def ramp(a: float, b: float, t0: float, t1: float) -> Callable[[float], float]:
    """Return a(t): a->b smoothly from t0..t1, else constant."""
    def fn(t: float) -> float:
        if t <= t0:
            return a
        if t >= t1:
            return b
        u = (t - t0) / (t1 - t0)
        u = smoothstep(u)
        return (1-u)*a + u*b
    return fn

def constant_force(vec=(3.0, 0.0, 1.5)) -> ForceFn:
    v = np.array(vec, dtype=float)
    def f(t: float, s: np.ndarray) -> np.ndarray:
        return v
    return f

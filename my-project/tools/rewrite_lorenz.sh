#!/usr/bin/env bash
set -e

# ---- lorenz simulator (RK4 + time-varying parameters + optional force) ----
cat > mylib/attractors/lorenz_sim.py <<'PY'
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

    # burn-in (no force)
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
PY

# ---- viz helpers (axes + side panel + segmented colored curve) ----
cat > mylib/attractors/lorenz_viz.py <<'PY'
from manim import *
import numpy as np

# Tutorial-ish dark background
config.background_color = "#06080f"

PALETTE = ["#ff4d6d", "#3a86ff", "#9b5de5", "#00f5d4", "#ffd166"]  # changes every 2s

def make_axes():
    # Larger ranges to hold a bigger (anesthesia) attractor
    ax = ThreeDAxes(
        x_range=[-60, 60, 20],
        y_range=[-70, 70, 20],
        z_range=[0, 120, 20],
        x_length=6.4,
        y_length=6.4,
        z_length=5.8,
        axis_config={"stroke_color": DARK_GRAY, "stroke_opacity": 0.55},
    )
    return ax

def side_panel(title: str, lines: list[str]):
    t = Text(title, font="Arial", font_size=32)
    t.set_color(WHITE)
    body = VGroup(*[Text(s, font="Arial", font_size=24).set_color(WHITE) for s in lines])
    body.arrange(DOWN, aligned_edge=LEFT, buff=0.18)
    panel = VGroup(t, body).arrange(DOWN, aligned_edge=LEFT, buff=0.28)
    panel.to_edge(RIGHT).shift(UP * 1.25)
    return panel

def segmented_curve(ax, pts_xyz: np.ndarray, dt: float, seg_seconds: float = 2.0, stroke_width: float = 3.0):
    """
    Every seg_seconds seconds => one VMobject with a fixed color.
    Already drawn segments keep their old color forever (what you want).
    """
    n_per = max(2, int(seg_seconds / dt))
    segments = []
    runtimes = []
    total = len(pts_xyz)
    start = 0
    color_idx = 0

    while start < total - 2:
        end = min(total, start + n_per + 1)
        chunk = pts_xyz[start:end]
        if len(chunk) < 2:
            break

        seg = VMobject(stroke_width=stroke_width)
        seg.set_points_smoothly([ax.c2p(x, y, z) for x, y, z in chunk])
        seg.set_color(PALETTE[color_idx % len(PALETTE)])
        segments.append(seg)

        dur = (len(chunk) - 1) * dt
        runtimes.append(max(dur, 1/15))  # avoid ultra-short warnings

        start = end - 1
        color_idx += 1

    return segments, runtimes

def shift_to_center(ax, segments: list[VMobject], extra_down: float = 0.35):
    """
    Compute a shift vector so the *trajectory itself* sits near screen center.
    We shift BOTH axes and segments together => no "colored line squeezed down".
    """
    traj = VGroup(*segments)
    shift = -traj.get_center() + DOWN * extra_down
    return shift
PY

# ---- Wake scene (NO static ghost curve; only time-drawn colored segments) ----
cat > mylib/scenes/LorenzWakeScene.py <<'PY'
from manim import *
import numpy as np
from mylib.attractors.lorenz_sim import simulate, constant_force
from mylib.attractors.lorenz_viz import make_axes, side_panel, segmented_curve, shift_to_center

class LorenzWakeScene(ThreeDScene):
    def construct(self):
        ax = make_axes()

        # Camera: zoom in a bit
        self.set_camera_orientation(phi=65 * DEGREES, theta=-45 * DEGREES, zoom=1.35)
        self.begin_ambient_camera_rotation(rate=0.10)

        panel = side_panel(
            "Wake state",
            [
                "Baseline (5 s)",
                "Perturbation (1 s) + deviation (5 s)",
                "Recovery to baseline (5 s)",
                "Fast resistance to perturbation",
            ],
        )
        self.add_fixed_in_frame_mobjects(panel)

        dt = 0.005

        # Make the wake attractor itself bigger by using a larger rho (not just zoom)
        rho_wake = 45.0
        sigma = 10.0
        beta = 8.0 / 3.0

        # Phase 1: baseline 5s
        pts1, s1 = simulate(s0=(0.1, 0.0, 0.0), duration=5.0, dt=dt, sigma=sigma, rho=rho_wake, beta=beta, burn_in=2.0)

        # Phase 2: deviation 5s under perturbation force
        force = constant_force(vec=(10.0, 0.0, 3.0))  # push outward
        pts2, s2 = simulate(s0=s1, duration=5.0, dt=dt, sigma=sigma, rho=rho_wake, beta=beta, burn_in=0.0, force=force)

        # Phase 3: recovery 5s (no force)
        pts3, _ = simulate(s0=s2, duration=5.0, dt=dt, sigma=sigma, rho=rho_wake, beta=beta, burn_in=0.0, force=None)

        pts_all = np.vstack([pts1, pts2, pts3])

        segments, runtimes = segmented_curve(ax, pts_all, dt=dt, seg_seconds=2.0, stroke_width=3.0)

        # A single "world transform": scale + shift for BOTH axes and segments
        world_scale = 1.10  # extra enlarge
        ax.scale(world_scale)
        for seg in segments:
            seg.scale(world_scale)

        shift = shift_to_center(ax, segments, extra_down=0.35)
        ax.shift(shift)
        for seg in segments:
            seg.shift(shift)

        self.add(ax)

        # Small vibration 1s at perturbation onset (jitter the whole axes a tiny bit)
        ax.save_state()
        self.wait(0.2)
        for _ in range(10):
            self.play(ax.animate.shift(0.03 * RIGHT), run_time=0.05, rate_func=linear)
            self.play(ax.animate.shift(0.03 * LEFT), run_time=0.05, rate_func=linear)
        self.play(Restore(ax), run_time=0.15)

        # Draw the colored trajectory over time (no static butterfly underneath)
        for seg, rt in zip(segments, runtimes):
            self.play(Create(seg), run_time=rt, rate_func=linear)

        self.wait(0.8)
PY

# ---- Anesthesia scene ----
cat > mylib/scenes/LorenzAnesthesiaScene.py <<'PY'
from manim import *
import numpy as np
from mylib.attractors.lorenz_sim import simulate, constant_force, ramp
from mylib.attractors.lorenz_viz import make_axes, side_panel, segmented_curve, shift_to_center

class LorenzAnesthesiaScene(ThreeDScene):
    def construct(self):
        ax = make_axes()

        self.set_camera_orientation(phi=65 * DEGREES, theta=-45 * DEGREES, zoom=1.35)
        self.begin_ambient_camera_rotation(rate=0.10)

        panel = side_panel(
            "Anesthesia state",
            [
                "Wake baseline (2 s)",
                "Anesthesia agent -> larger attractor (5 s)",
                "Perturbation (1 s) + deviation (10 s)",
                "Recovery longer than wake",
                "Return to wake baseline",
            ],
        )
        self.add_fixed_in_frame_mobjects(panel)

        dt = 0.005

        rho_wake = 45.0
        rho_anes = 65.0  # bigger butterfly by equation parameter
        sigma = 10.0
        beta = 8.0 / 3.0

        # A) wake baseline 2s
        ptsA, sA = simulate(s0=(0.1, 0.0, 0.0), duration=2.0, dt=dt, sigma=sigma, rho=rho_wake, beta=beta, burn_in=2.0)

        # B) "agent" phase: 1s smooth ramp wake->anes + 4s hold => total 5s
        rho_ramp = ramp(rho_wake, rho_anes, t0=0.0, t1=1.0)
        ptsB1, sB1 = simulate(s0=sA, duration=1.0, dt=dt, sigma=sigma, rho=rho_ramp, beta=beta, burn_in=0.0)
        ptsB2, sB2 = simulate(s0=sB1, duration=4.0, dt=dt, sigma=sigma, rho=rho_anes, beta=beta, burn_in=0.0)
        ptsB = np.vstack([ptsB1, ptsB2])

        # C) deviation 10s under perturbation force (stronger/longer)
        force = constant_force(vec=(10.0, 0.0, 3.0))
        ptsC, sC = simulate(s0=sB2, duration=10.0, dt=dt, sigma=sigma, rho=rho_anes, beta=beta, burn_in=0.0, force=force)

        # D) recovery 5s
        ptsD, sD = simulate(s0=sC, duration=5.0, dt=dt, sigma=sigma, rho=rho_anes, beta=beta, burn_in=0.0, force=None)

        # E) return to wake: 1s ramp back + 2s hold
        rho_back = ramp(rho_anes, rho_wake, t0=0.0, t1=1.0)
        ptsE1, sE1 = simulate(s0=sD, duration=1.0, dt=dt, sigma=sigma, rho=rho_back, beta=beta, burn_in=0.0)
        ptsE2, _ = simulate(s0=sE1, duration=2.0, dt=dt, sigma=sigma, rho=rho_wake, beta=beta, burn_in=0.0)
        ptsE = np.vstack([ptsE1, ptsE2])

        pts_all = np.vstack([ptsA, ptsB, ptsC, ptsD, ptsE])

        segments, runtimes = segmented_curve(ax, pts_all, dt=dt, seg_seconds=2.0, stroke_width=3.0)

        world_scale = 1.10
        ax.scale(world_scale)
        for seg in segments:
            seg.scale(world_scale)

        shift = shift_to_center(ax, segments, extra_down=0.35)
        ax.shift(shift)
        for seg in segments:
            seg.shift(shift)

        self.add(ax)

        # Vibration 1s at perturbation onset
        ax.save_state()
        self.wait(0.2)
        for _ in range(10):
            self.play(ax.animate.shift(0.03 * RIGHT), run_time=0.05, rate_func=linear)
            self.play(ax.animate.shift(0.03 * LEFT), run_time=0.05, rate_func=linear)
        self.play(Restore(ax), run_time=0.15)

        for seg, rt in zip(segments, runtimes):
            self.play(Create(seg), run_time=rt, rate_func=linear)

        self.wait(0.8)
PY

echo "Done writing files."

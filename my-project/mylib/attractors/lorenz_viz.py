from manim import *
import numpy as np

config.background_color = "#06080f"
PALETTE = ["#ff4d6d", "#3a86ff", "#9b5de5", "#00f5d4", "#ffd166"]

def make_axes():
    ax = ThreeDAxes(
        x_range=[-60, 60, 20],
        y_range=[-70, 70, 20],
        z_range=[0, 110, 20],
        x_length=6.2,
        y_length=6.2,
        z_length=5.8,
        axis_config={"stroke_color": DARK_GRAY, "stroke_opacity": 0.35, "stroke_width": 2},
    )
    return ax

def equation_block():
    rho = MathTex(r"\rho=\rho(t)", font_size=44).set_color("#ffd166")
    eq1 = MathTex(r"\dot{x}=\sigma (y-x)", font_size=38).set_color("#ff4d6d")
    eq2 = MathTex(r"\dot{y}=x(\rho-z)-y", font_size=38).set_color("#3a86ff")
    eq3 = MathTex(r"\dot{z}=xy-\beta z+u(t)", font_size=38).set_color("#00f5d4")
    block = VGroup(rho, eq1, eq2, eq3).arrange(DOWN, aligned_edge=LEFT, buff=0.16)
    block.to_corner(UL).shift(RIGHT * 0.45 + DOWN * 0.35)
    return block

def title_tag(text: str):
    t = Text(text, font_size=36, weight=BOLD, color=WHITE)
    t.to_corner(UR).shift(LEFT * 0.45 + DOWN * 0.35)
    return t

def watermark():
    w = Text("@绝命麻师  @Yousafchao", font_size=52, weight=BOLD)
    w.set_opacity(0.33)
    w.move_to(ORIGIN)
    return w

def acknowledgement():
    a = Text(
        "Ack.Inspired by Earl K. Miller et al., 2024, Neuron.",
        font_size=20,
        color=WHITE,
    )
    a.set_opacity(0.55)
    a.to_edge(DOWN).shift(UP * 0.25)
    return a

def conclusion_box():
    lines = [
        "Compared with the conscious state, anesthetic agents can shift the brain",
        "into a regime of greater dynamical complexity and instability,",
        "making neural activity more vulnerable to internal and external perturbations.",
    ]
    body = Paragraph(*lines, alignment="left", font_size=20, line_spacing=0.7)
    body.set_opacity(0.85)

    box = RoundedRectangle(
        corner_radius=0.18,
        width=body.width + 0.6,
        height=body.height + 0.5,
        stroke_color=WHITE,
        stroke_opacity=0.30,
        fill_color=BLACK,
        fill_opacity=0.35,
    )
    panel = VGroup(box, body)
    body.move_to(box.get_center()).shift(LEFT * 0.05)

    panel.to_corner(DL).shift(RIGHT * 0.55 + UP * 0.95)
    return panel

def status_box(current: str, include_agent: bool):
    items = ["Baseline"]
    if include_agent:
        items.append("Anesthesia Agent")
    items.append("Perturbation")

    lines = []
    for s in items:
        if s == current:
            lines.append(Text(s, font_size=34, weight=BOLD, color=YELLOW))
        else:
            lines.append(Text(s, font_size=30, color=WHITE).set_opacity(0.55))

    body = VGroup(*lines).arrange(DOWN, aligned_edge=LEFT, buff=0.14)

    box = RoundedRectangle(
        corner_radius=0.18,
        width=body.width + 0.6,
        height=body.height + 0.5,
        stroke_color=WHITE,
        stroke_opacity=0.35,
        fill_color=BLACK,
        fill_opacity=0.35,
    )
    panel = VGroup(box, body)
    body.move_to(box.get_center()).shift(LEFT * 0.05)

    panel.to_corner(DR).shift(LEFT * 0.55 + UP * 1.15)
    return panel

def shake_animation(mob: Mobject, amp: float = 0.45, shakes: int = 14, run_time: float = 1.2, seed: int = 1):
    rng = np.random.default_rng(seed)
    per = run_time / shakes
    anims = []
    for _ in range(shakes):
        v = rng.normal(size=2)
        v = v / (np.linalg.norm(v) + 1e-9)
        vec = np.array([v[0], v[1], 0.0]) * amp
        anims.append(
            mob.animate.shift(vec).set_rate_func(there_and_back).set_run_time(per)
        )
    return Succession(*anims)

def segmented_curve(
    ax,
    pts_xyz: np.ndarray,
    dt: float,
    seg_seconds: float = 20.0,
    stroke_width: float = 3.0,
    display_scale: float = 1.0,
    extra_shift=ORIGIN,
):
    n_per = max(2, int(seg_seconds / dt))
    segments, runtimes = [], []
    total = len(pts_xyz)
    start = 0
    color_idx = 0

    while start < total - 2:
        end = min(total, start + n_per + 1)
        chunk = pts_xyz[start:end]
        if len(chunk) < 2:
            break

        seg = VMobject(stroke_width=stroke_width)
        seg.set_points_smoothly(
            [ax.c2p(x*display_scale, y*display_scale, z*display_scale) for x, y, z in chunk]
        )
        seg.set_color(PALETTE[color_idx % len(PALETTE)])
        seg.shift(extra_shift)

        segments.append(seg)
        runtimes.append((len(chunk) - 1) * dt)

        start = end - 1
        color_idx += 1

    return segments, runtimes

def center_shift_from_points(ax, pts_xyz: np.ndarray, display_scale: float = 1.0):
    sample = pts_xyz[::10] if len(pts_xyz) > 50 else pts_xyz
    pts = np.array([ax.c2p(x*display_scale, y*display_scale, z*display_scale) for x, y, z in sample])
    center = pts.mean(axis=0)
    return -center + DOWN * 0.25

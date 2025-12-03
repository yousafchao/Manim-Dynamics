from manim import *
import numpy as np

# 教程风格：深背景
config.background_color = "#06080f"

PALETTE = ["#ff4d6d", "#3a86ff", "#9b5de5", "#00f5d4", "#ffd166"]  # 每2秒换一次

def make_axes():
    # 范围给得稍大：能容纳更大(麻醉)吸引子
    ax = ThreeDAxes(
        x_range=[-45, 45, 15],
        y_range=[-55, 55, 20],
        z_range=[0, 75, 20],
        x_length=6.4,
        y_length=6.4,
        z_length=5.6,
        axis_config={"stroke_color": DARK_GRAY, "stroke_opacity": 0.55},
    )
    return ax

def side_panel(title: str, lines: list[str]):
    t = Text(title, font="Calibri", font_size=32, weight=BOLD)
    body = VGroup(*[Text(s, font="Arial", font_size=22) for s in lines]).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
    panel = VGroup(t, body).arrange(DOWN, aligned_edge=LEFT, buff=0.28)
    panel.to_edge(RIGHT).shift(UP*1.25)
    return panel

def segmented_curve(ax, pts_xyz: np.ndarray, dt: float, seg_seconds: float = 2.0, stroke_width: float = 3.0):
    """
    每 seg_seconds 秒生成一段 VMobject，给固定颜色（旧轨迹不变）。
    返回 segments(list[VMobject])，以及每段对应的建议 run_time。
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

        # 最后一段可能不足 seg_seconds
        dur = (len(chunk) - 1) * dt
        runtimes.append(dur)

        start = end - 1
        color_idx += 1

    return segments, runtimes

def center_shift_from_points(ax, pts_xyz: np.ndarray):
    """
    用轨迹重心把“蝴蝶”移到画面中心附近（不是镜头zoom，是整体平移）。
    """
    sample = pts_xyz[::10] if len(pts_xyz) > 50 else pts_xyz
    pts = np.array([ax.c2p(x, y, z) for x, y, z in sample])
    center = pts.mean(axis=0)
    # 额外 DOWN 一点防止 z 轴顶到屏幕上沿
    return -center + DOWN*0.2

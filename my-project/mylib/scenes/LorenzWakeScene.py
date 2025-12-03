from manim import *
import numpy as np

from mylib.systems.lorenz import simulate, constant_force
from mylib.attractors.lorenz_viz import (
    conclusion_box,
    shake_animation,
    make_axes, equation_block, title_tag, watermark, acknowledgement,
    status_box, segmented_curve, center_shift_from_points
)

def park_snapshot(mob: Mobject, label: str, offset: np.ndarray):
    tag = Text(label, font_size=28, weight=BOLD, color=WHITE).set_opacity(0.85)
    tag.next_to(mob, DOWN, buff=0.25)
    return VGroup(mob, tag).shift(offset)

class LorenzWakeScene(ThreeDScene):
    def construct(self):
        DT = 0.01
        COLOR_PERIOD = 20.0

        BASELINE_T = 50.0
        PERT_PAUSE = 10.0
        DEVIATE_T = 50.0
        RECOVER_T = 50.0

        SIGMA = 10.0
        BETA = 8.0 / 3.0
        RHO_WAKE = 40.0

        display_scale = 1.0

        ax = make_axes()
        self.set_camera_orientation(phi=65 * DEGREES, theta=-45 * DEGREES, zoom=1.25)
        self.begin_ambient_camera_rotation(rate=0.10)

        # --- 固定在屏幕上的 UI（不跟镜头转）---
        eq = equation_block()
        title = title_tag("Wake state")
        w = watermark()
        ack = acknowledgement()
        concl = conclusion_box()
        hud = status_box("Baseline", include_agent=False)

        self.add(ax)

        # 关键：fixed 之后也要 add，否则经常“你以为有但看不见”
        self.add_fixed_in_frame_mobjects(eq, title, w, ack, hud, concl)
        self.add(eq, title, w, ack, hud, concl)

        # --- Baseline ---
        ptsA, sA = simulate(
            s0=(0.0, 1.0, 1.05),
            duration=BASELINE_T,
            dt=DT,
            sigma=SIGMA,
            rho=RHO_WAKE,
            beta=BETA,
            burn_in=2.0,
            force=None,
        )
        shift_center = center_shift_from_points(ax, ptsA, display_scale=display_scale)

        segA, runA = segmented_curve(
            ax, ptsA, DT, seg_seconds=COLOR_PERIOD,
            display_scale=display_scale, extra_shift=shift_center
        )
        grpA = VGroup()
        for seg, rt in zip(segA, runA):
            self.play(Create(seg), run_time=rt, rate_func=linear)
            grpA.add(seg)

        # --- “Baseline -> Perturbation” 平移入档（上一版效果）---
        R = 4.0
        SLOT_180 = LEFT * R
        SLOT_0 = RIGHT * R

        new_hud = status_box("Perturbation", include_agent=False)
        parkedA = park_snapshot(grpA, "Baseline", offset=SLOT_180)

        self.play(
            AnimationGroup(
                Transform(hud, new_hud),
                Transform(grpA, parkedA[0]),
                FadeIn(parkedA[1], shift=DOWN * 0.05),
                lag_ratio=0,
            ),
            shake_animation(ax, run_time=1.8),
        )
        self.wait(max(0.0, PERT_PAUSE - 1.8))

        # --- Deviation (force) ---
        force = constant_force(vec=(3.0, 0.0, 1.5))
        ptsB, sB = simulate(
            s0=sA,
            duration=DEVIATE_T,
            dt=DT,
            sigma=SIGMA,
            rho=RHO_WAKE,
            beta=BETA,
            burn_in=0.0,
            force=force,
        )
        segB, runB = segmented_curve(
            ax, ptsB, DT, seg_seconds=COLOR_PERIOD,
            display_scale=display_scale, extra_shift=shift_center
        )
        grpB = VGroup()
        for seg, rt in zip(segB, runB):
            self.play(Create(seg), run_time=rt, rate_func=linear)
            grpB.add(seg)

        # --- “Perturbation -> Baseline” 平移入档 ---
        new_hud2 = status_box("Baseline", include_agent=False)
        parkedB = park_snapshot(grpB, "Perturbation", offset=SLOT_0)

        self.play(
            AnimationGroup(
                Transform(hud, new_hud2),
                Transform(grpB, parkedB[0]),
                FadeIn(parkedB[1], shift=DOWN * 0.05),
                lag_ratio=0,
            ),
            shake_animation(ax, run_time=1.8),
        )

        # --- Recovery ---
        ptsC, _ = simulate(
            s0=sB,
            duration=RECOVER_T,
            dt=DT,
            sigma=SIGMA,
            rho=RHO_WAKE,
            beta=BETA,
            burn_in=0.0,
            force=None,
        )
        segC, runC = segmented_curve(
            ax, ptsC, DT, seg_seconds=COLOR_PERIOD,
            display_scale=display_scale, extra_shift=shift_center
        )
        for seg, rt in zip(segC, runC):
            self.play(Create(seg), run_time=rt, rate_func=linear)

        self.wait(1)

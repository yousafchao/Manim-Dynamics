from manim import *
from mylib.systems.lorenz import simulate, ramp, constant_force
from mylib.attractors.lorenz_viz import (
    conclusion_box,
    shake_animation,
    make_axes,
    equation_block,
    title_tag,
    watermark,
    acknowledgement,
    status_box,
    segmented_curve,
    center_shift_from_points,
)
def park_snapshot(mob: Mobject, label: str, offset: np.ndarray):
    tag = Text(label, font_size=28, weight=BOLD, color=WHITE).set_opacity(0.85)
    tag.next_to(mob, DOWN, buff=0.25)
    return VGroup(mob, tag).shift(offset)

class LorenzAnesthesiaScene(ThreeDScene):
    def construct(self):
        hud = status_box("Baseline", include_agent=True)
        concl = conclusion_box()
        self.add_fixed_in_frame_mobjects(hud, concl)
        self.add(hud, concl)
        DT = 0.01
        COLOR_PERIOD = 20.0

        BASELINE_PRE = 20.0
        AGENT_T = 50.0
        PERT_PAUSE = 10.0
        DEVIATE_T = 100.0
        RECOVER_T = 50.0

        SIGMA = 10.0
        BETA = 8.0 / 3.0
        RHO_WAKE = 40.0
        RHO_ANES = 65.0

        display_scale = 0.5

        ax = make_axes()
        self.set_camera_orientation(phi=65 * DEGREES, theta=-45 * DEGREES, zoom=1.25)
        self.begin_ambient_camera_rotation(rate=0.10)

        eq = equation_block()
        title = title_tag("Anesthesia state")
        w = watermark()
        ack = acknowledgement()

        self.add(ax)
        self.add_fixed_in_frame_mobjects(title, eq, w, ack)
        self.add(title, eq, w, ack)

        ptsA, sA = simulate(
            s0=(0.0, 1.0, 1.05),
            duration=BASELINE_PRE,
            dt=DT,
            sigma=SIGMA,
            rho=RHO_WAKE,
            beta=BETA,
            burn_in=2.0,
            force=None,
        )
        shift_center = center_shift_from_points(ax, ptsA, display_scale=display_scale)

        segA, runA = segmented_curve(ax, ptsA, DT, seg_seconds=COLOR_PERIOD, display_scale=display_scale, extra_shift=shift_center)
        grpA = VGroup()
        for seg, rt in zip(segA, runA):
            self.play(Create(seg), run_time=rt, rate_func=linear)
            grpA.add(seg)

        R = 3.2
        SLOT_180 = LEFT * R
        SLOT_90 = UP * R
        SLOT_270 = DOWN * R

        new_hud_agent = status_box("Anesthesia Agent", include_agent=True)
        parkedA = park_snapshot(grpA, "Baseline", offset=SLOT_180)

        self.play(
            AnimationGroup(
                Transform(hud, new_hud_agent),
                Transform(grpA, parkedA[0]),
                FadeIn(parkedA[1], shift=DOWN * 0.05),
                lag_ratio=0,
            ),
            shake_animation(ax, run_time=1.8),
        )

        rho_fn = ramp(RHO_WAKE, RHO_ANES, t0=0.0, t1=10.0)
        ptsB, sB = simulate(
            s0=sA,
            duration=AGENT_T,
            dt=DT,
            sigma=SIGMA,
            rho=rho_fn,
            beta=BETA,
            burn_in=0.0,
            force=None,
        )
        segB, runB = segmented_curve(ax, ptsB, DT, seg_seconds=COLOR_PERIOD, display_scale=display_scale, extra_shift=shift_center)
        grpB = VGroup()
        for seg, rt in zip(segB, runB):
            self.play(Create(seg), run_time=rt, rate_func=linear)
            grpB.add(seg)

        new_hud_pert = status_box("Perturbation", include_agent=True)
        parkedB = park_snapshot(grpB, "Anesthesia Agent", offset=SLOT_90)

        self.play(
            AnimationGroup(
                Transform(hud, new_hud_pert),
                ApplyMethod(ax.shift, RIGHT * 0.14, rate_func=there_and_back),
                Transform(grpB, parkedB[0]),
                FadeIn(parkedB[1], shift=DOWN * 0.05),
                lag_ratio=0,
            ),
            shake_animation(ax, run_time=1.8),
        )

        self.wait(max(0.0, PERT_PAUSE - 1.0))

        force = constant_force(vec=(3.0, 0.0, 1.5))
        ptsC, sC = simulate(
            s0=sB,
            duration=DEVIATE_T,
            dt=DT,
            sigma=SIGMA,
            rho=RHO_ANES,
            beta=BETA,
            burn_in=0.0,
            force=force,
        )
        segC, runC = segmented_curve(ax, ptsC, DT, seg_seconds=COLOR_PERIOD, display_scale=display_scale, extra_shift=shift_center)
        grpC = VGroup()
        for seg, rt in zip(segC, runC):
            self.play(Create(seg), run_time=rt, rate_func=linear)
            grpC.add(seg)

        new_hud_base = status_box("Baseline", include_agent=True)
        parkedC = park_snapshot(grpC, "Perturbation", offset=SLOT_270)

        self.play(
            AnimationGroup(
                Transform(hud, new_hud_base),
                Transform(grpC, parkedC[0]),
                FadeIn(parkedC[1], shift=DOWN * 0.05),
                lag_ratio=0,
            ),
            shake_animation(ax, run_time=1.8),
        )

        rho_back = ramp(RHO_ANES, RHO_WAKE, t0=0.0, t1=10.0)
        ptsD, _ = simulate(
            s0=sC,
            duration=RECOVER_T,
            dt=DT,
            sigma=SIGMA,
            rho=rho_back,
            beta=BETA,
            burn_in=0.0,
            force=None,
        )
        segD, runD = segmented_curve(ax, ptsD, DT, seg_seconds=COLOR_PERIOD, display_scale=display_scale, extra_shift=shift_center)
        grpD = VGroup()
        for seg, rt in zip(segD, runD):
            self.play(Create(seg), run_time=rt, rate_func=linear)
            grpD.add(seg)
        # === Finale: “freecell weave” interlacing ===
        # 把已有的几个停放曲线拿出来（只取曲线，不要下面的小标签文字）
        curves = VGroup(parkedA[0], parkedB[0], parkedC[0], grpD)

        # 可选：把停放标签淡出，让画面更干净
        # self.play(FadeOut(parkedA[1], parkedB[1], parkedC[1]), run_time=0.6)

        # 让坐标轴淡一点，让“编织”更像主角
        self.play(ax.animate.set_opacity(0.25), run_time=0.8)

        # 把所有曲线先收拢到中心附近（注意是整体 move，不是改变方程）
        center = ORIGIN + DOWN * 0.15
        offsets = [LEFT * 0.9, RIGHT * 0.9, UP * 0.9, DOWN * 0.9]
        for i, c in enumerate(curves):
            c.generate_target()
            c.target.move_to(center + offsets[i % len(offsets)] + (i - 1.5) * 0.20 * OUT)  # OUT 做景深错层
            c.target.scale(1.12)
        self.play(*[MoveToTarget(c) for c in curves], run_time=2.2)

        # 用 ValueTracker 驱动“环绕”，每条曲线相位不同 + 自旋速度不同
        t_weave = ValueTracker(0.0)
        radius = 1.15
        n = len(curves)

        def make_weave_updater(i: int):
            phase = i * TAU / n
            spin = 0.55 + 0.12 * i
            zoff = (i - (n - 1) / 2) * 0.22

            def _upd(mob, dt):
                ang = t_weave.get_value() + phase
                mob.move_to(center + radius * np.cos(ang) * RIGHT + radius * np.sin(ang) * UP + zoff * OUT)
                mob.rotate(spin * dt, axis=OUT, about_point=center)

            return _upd

        for i, c in enumerate(curves):
            c.add_updater(make_weave_updater(i))

        # 让它织一会儿；你也可以在这段期间重新开启轻微镜头旋转
        # self.begin_ambient_camera_rotation(rate=0.06)

        self.play(t_weave.animate.set_value(6 * TAU), run_time=10, rate_func=linear)

        for c in curves:
            c.clear_updaters()

        self.play(ax.animate.set_opacity(0.55), run_time=0.6)

        self.wait(1)

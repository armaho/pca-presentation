import random

from manim import *


class CovarianceExplanation(Scene):
    def construct(self):
        text = Text("Covariance").scale(1.5)

        formula1 = MathTex(r"\frac{1}{n} \sum_{i=1}^{n} (X_i - \overline{X})(Y_i - \overline{Y})")

        self.play(FadeIn(text))
        self.play(Transform(text, formula1))
        self.play(text.animate.to_edge(RIGHT, buff=1))

        grid = NumberPlane(
            x_range=[-5, 5, 1],
            y_range=[-5, 5, 1],
            x_length=6,
            y_length=6,
            axis_config={"color": BLUE},
            background_line_style={
                "stroke_color": WHITE,
                "stroke_width": 1,
                "stroke_opacity": 0.5,
            },
        )

        grid.to_edge(LEFT, buff=0.5)

        x_label = MathTex("x").next_to(grid.x_axis.get_end(), RIGHT)
        y_label = MathTex("y").next_to(grid.y_axis.get_end(), UP)

        self.play(FadeIn(grid), Create(x_label), Create(y_label))

        points_cnt = 60
        points_data_trackers = [(ValueTracker(random.uniform(-2, 5)), ValueTracker(random.uniform(-5, 2))) for _ in range(points_cnt)]

        dots = VGroup()
        for x_tracker, y_tracker in points_data_trackers:
            def dot_updater(d, xt=x_tracker, yt=y_tracker):
                x, y = xt.get_value(), yt.get_value()
                eps = 0.01

                x_mean = sum([x.get_value() for x, _ in points_data_trackers]) / points_cnt
                y_mean = sum([y.get_value() for _, y in points_data_trackers]) / points_cnt

                if -eps <= x_mean <= eps and -eps <= y_mean <= eps:
                    value = ((x - x_mean) * (y - y_mean) / 4) + 0.5
                    d.set_color(interpolate_color(RED, GREEN, value))

                d.move_to(grid.c2p(x, y))

            dot = Dot(point=[x_tracker.get_value(), y_tracker.get_value(), 0], color=RED).add_updater(dot_updater)

            dots.add(dot)

        self.play(FadeIn(dots))

        x_mean = sum([x.get_value() for x, _ in points_data_trackers]) / points_cnt
        y_mean = sum([y.get_value() for _, y in points_data_trackers]) / points_cnt

        self.play(*[x_tracker.animate.increment_value(-1 * x_mean) for x_tracker, _ in points_data_trackers], run_time=2)
        self.wait(2)

        formula2 = MathTex(r"\frac{1}{n} \sum_{i=1}^{n} X_i(Y_i - \overline{Y})")
        formula2.move_to(text)

        self.play(FadeOut(text), FadeIn(formula2))

        self.play(*[y_tracker.animate.increment_value(-1 * y_mean) for _, y_tracker in points_data_trackers], run_time=2)
        self.wait(2)

        text.shift(LEFT)

        def calc_cov():
            x_mean = sum([x.get_value() for x, _ in points_data_trackers]) / points_cnt
            y_mean = sum([y.get_value() for _, y in points_data_trackers]) / points_cnt

            return sum([(x.get_value() - x_mean) * (y.get_value() - y_mean) for x, y in points_data_trackers]) / points_cnt

        formula4 = MathTex(r"\frac{1}{n} \sum_{i=1}^{n} X_iY_i")
        formula4.move_to(formula2)

        self.play(FadeOut(formula2), FadeIn(formula4))
        self.wait()

        formula3 = MathTex(r"\frac{1}{n} \sum_{i=1}^{n} X_iY_i = ")

        cov = DecimalNumber(calc_cov(), num_decimal_places=2)
        cov.add_updater(lambda t: t.set_value(calc_cov()))

        equation = VGroup(formula3, cov).arrange(RIGHT)

        equation.move_to(formula4)

        self.play(FadeOut(formula4), FadeIn(equation))
        self.wait()

        self.play(*[y_tracker.animate.set_value(0.75 * x_tracker.get_value() + 0.25 * y_tracker.get_value()) for x_tracker, y_tracker in points_data_trackers], run_time=5)

        y_mean = sum([y.get_value() for _, y in points_data_trackers]) / points_cnt
        self.play(*[y_tracker.animate.increment_value(-1 * y_mean) for _, y_tracker in points_data_trackers])

        self.wait(2)


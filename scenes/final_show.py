import math

from manim import *


class FinalShowScene(Scene):
    def construct(self):
        plane = NumberPlane(
            x_range=[-8, 8, 1],
            y_range=[-8, 8, 1],
            x_length=6,
            y_length=6,
            axis_config={
                "color": BLUE,
                "stroke_width": 2,
            },
            background_line_style={
                "stroke_color": WHITE,
                "stroke_width": 1,
            },
        )

        x_label = MathTex("x").next_to(plane.x_axis.get_end(), RIGHT)
        y_label = MathTex("y").next_to(plane.y_axis.get_end(), UP)

        print(plane.coords_to_point((1, 0))[:2])

        i = Vector([0.375, 0], color=GREEN_D, max_tip_length_to_length_ratio=0, stroke_width=20)
        j = Vector([0, 0.375], color=RED, max_tip_length_to_length_ratio=0, stroke_width=20)

        self.play(FadeIn(plane), Write(x_label), Write(y_label))
        self.play(Create(i), Create(j))

        point_cnt = 150
        random_coords = np.random.normal(0, 1, (point_cnt, 2)).tolist()
        points = [plane.c2p(x, y) for x, y in random_coords]
        dots_1 = [Dot(point, color=YELLOW, radius=DEFAULT_DOT_RADIUS / 2) for point in points]

        self.play(FadeIn(*dots_1))
        self.wait()

        dots_2 = [Dot([point[0], point[1], point[2]], color=YELLOW, radius=DEFAULT_DOT_RADIUS / 2) for point in points]
        dots_3 = [Dot([point[0], 3 * point[1], point[2]], color=YELLOW, radius=DEFAULT_DOT_RADIUS / 2) for point in points]

        self.play(*[ReplacementTransform(dots_1[i], dots_2[i]) for i in range(point_cnt)])
        self.wait()

        self.play(*[ReplacementTransform(dots_2[i], dots_3[i]) for i in range(point_cnt)])
        self.wait()

        blah = math.sqrt(2)/2

        dots_4 = [Dot([blah * point[0] - 3 * blah * point[1], blah * point[0] + 3 * blah * point[1], point[2]], color=YELLOW, radius=DEFAULT_DOT_RADIUS / 2) for point in points]
        self.play(*[ReplacementTransform(dots_3[i], dots_4[i]) for i in range(point_cnt)])
        self.wait()


if __name__ == "__main__":
    point_cnt = 40
    random_coords = np.random.normal(0, 1, (point_cnt, 2)).tolist()

    print(random_coords[:2])

import math

from manim import *


def create_matrix_with_box(matrix) -> VGroup:
    m = Matrix(matrix)
    box = Rectangle(
        width=m.width + 0.5,
        height=m.height + 0.5,
        color=BLACK,
        fill_opacity=0.5,
        stroke_width=0
    ).move_to(m.get_center())

    return VGroup(box, m)


def create_math_text_with_background(text) -> VGroup:
    text = MathTex(text)
    bg = BackgroundRectangle(text, color=BLACK, fill_opacity=0.5, buff=0.2).move_to(text.center())

    return VGroup(bg, text)


class LinearTransformationsExampleScene(VectorScene):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.background_plane_kwargs = {
            "color": GREY,
            "axis_config": {
                "color": GREY,
            },
            "background_line_style": {
                "stroke_color": GREY,
                "stroke_width": 1,
            },
        }
        self.background_plane = NumberPlane(**self.background_plane_kwargs)
        self.add(self.background_plane)
        self.plane = self.add_plane()

    def construct(self):
        m_entries = [[2, 1], [0, 3]]
        m_2_entries = [[1, 0], [1, 1]]
        m_inverse_entries = [[2 / 3, -1 / 6], [-1 / 3, 1 / 3]]

        self.play(self.plane.animate.apply_matrix(m_entries), run_time=2)
        self.wait(1)
        self.play(self.plane.animate.apply_matrix(m_2_entries), run_time=2)
        self.wait(1)
        self.play(self.plane.animate.apply_matrix(m_inverse_entries), run_time=2)
        self.wait(1)

        i = Vector([1, 0], color=GREEN_D, max_tip_length_to_length_ratio=0)
        j = Vector([0, 1], color=RED, max_tip_length_to_length_ratio=0)
        u = Vector([2, 1], color=LIGHT_BROWN, max_tip_length_to_length_ratio=0)

        for vector in [i, j, u]:
            vector.set_max_tip_length_to_length_ratio(0.1)

        i_label = MathTex(r"\hat{i}", color=GREEN_D).add_updater(lambda t: t.next_to(i.get_end(), DOWN))
        j_label = MathTex(r"\hat{j}", color=RED).add_updater(lambda t: t.next_to(j.get_end(), RIGHT))
        u_label = MathTex(r"\hat{u} = 2\hat{i} + \hat{j}", color=LIGHT_BROWN).add_updater(
            lambda t: t.next_to(u.get_end(), RIGHT))

        self.play(Create(i), Create(j), Create(i_label), Create(j_label))
        self.play(Create(u), Create(u_label))

        m_3_entries = [[1, 1], [-1, 2]]

        self.play(self.plane.animate.apply_matrix(m_3_entries), i.animate.apply_matrix(m_3_entries),
                  j.animate.apply_matrix(m_3_entries), u.animate.apply_matrix(m_3_entries), run_time=2)

        matrix = Matrix(m_3_entries)
        box = Rectangle(
            width=matrix.width + 0.5,
            height=matrix.height + 0.5,
            color=BLACK,
            fill_opacity=0.5,
            stroke_width=0
        ).move_to(matrix.get_center())

        matrix_group = VGroup(box, matrix)
        matrix_group.to_corner(UL)

        self.play(FadeIn(matrix_group))
        self.wait()


class SomeMoreLinearTransformationExamples(Scene):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.background_plane_kwargs = {
            "color": GREY,
            "axis_config": {
                "color": GREY,
            },
            "background_line_style": {
                "stroke_color": GREY,
                "stroke_width": 1,
            },
        }
        self.background_plane = NumberPlane(**self.background_plane_kwargs)
        self.add(self.background_plane)
        self.plane = NumberPlane(x_range=[-10, 10, 1], y_range=[-10, 10, 1])
        self.add(self.plane)

    def construct(self):
        i = Vector([1, 0], color=GREEN_D, max_tip_length_to_length_ratio=0)
        j = Vector([0, 1], color=RED, max_tip_length_to_length_ratio=0)

        i_label = MathTex(r"\hat{i}", color=GREEN_D).add_updater(lambda t: t.next_to(i.get_end(), DOWN))
        j_label = MathTex(r"\hat{j}", color=RED).add_updater(lambda t: t.next_to(j.get_end(), RIGHT))

        self.add(i, j, i_label, j_label)

        m_1_entries = [[1, 1],
                       [0, 1]]
        m_1_inverse_entries = [[1, -1],
                               [0, 1]]
        m_1_matrix = Matrix(m_1_entries)
        m_1_box = Rectangle(
            width=m_1_matrix.width + 0.5,
            height=m_1_matrix.height + 0.5,
            color=BLACK,
            fill_opacity=0.5,
            stroke_width=0
        ).move_to(m_1_matrix.get_center())
        m_1 = VGroup(m_1_box, m_1_matrix)

        m_1.to_corner(UL)
        self.play(FadeIn(m_1))

        self.wait(2)
        self.play(self.plane.animate.apply_matrix(m_1_entries), i.animate.apply_matrix(m_1_entries),
                  j.animate.apply_matrix(m_1_entries), run_time=2)
        self.wait(2)
        self.play(self.plane.animate.apply_matrix(m_1_inverse_entries), i.animate.apply_matrix(m_1_inverse_entries),
                  j.animate.apply_matrix(m_1_inverse_entries), run_time=2)

        self.remove(m_1)

        m_2_entries = [[0, -1],
                       [1, 0]]
        m_2_inverse_entries = [[0, 1],
                               [-1, 0]]
        m_2_matrix = Matrix(m_2_entries)
        m_2_box = Rectangle(
            width=m_2_matrix.width + 0.5,
            height=m_2_matrix.height + 0.5,
            color=BLACK,
            fill_opacity=0.5,
            stroke_width=0
        ).move_to(m_2_matrix.get_center())
        m_2 = VGroup(m_2_box, m_2_matrix)

        m_2.to_corner(UL)
        self.play(FadeIn(m_2))

        self.wait(2)
        self.play(self.plane.animate.apply_matrix(m_2_entries), i.animate.apply_matrix(m_2_entries),
                  j.animate.apply_matrix(m_2_entries), run_time=2)
        self.wait(2)
        self.play(self.plane.animate.apply_matrix(m_2_inverse_entries), i.animate.apply_matrix(m_2_inverse_entries),
                  j.animate.apply_matrix(m_2_inverse_entries), run_time=2)

        self.remove(m_2)

        m_3_entries = [[2, 0],
                       [0, 3]]
        m_3_inverse = [[1 / 2, 0],
                       [0, 1 / 3]]
        m_3_matrix = Matrix(m_3_entries)
        m_3_box = Rectangle(
            width=m_2_matrix.width + 0.5,
            height=m_2_matrix.height + 0.5,
            color=BLACK,
            fill_opacity=0.5,
            stroke_width=0
        ).move_to(m_3_matrix.get_center())
        m_3 = VGroup(m_3_box, m_3_matrix)

        m_3.to_corner(UL)
        self.play(FadeIn(m_3))

        self.wait(2)
        self.play(self.plane.animate.apply_matrix(m_3_entries), i.animate.apply_matrix(m_3_entries),
                  j.animate.apply_matrix(m_3_entries), run_time=2)
        self.wait(2)
        self.play(self.plane.animate.apply_matrix(m_3_inverse), i.animate.apply_matrix(m_3_inverse),
                  j.animate.apply_matrix(m_3_inverse), run_time=2)

        self.remove(m_3)

        m_4_entries = [[4, 2],
                       [2, 1]]
        m_4 = create_matrix_with_box(m_4_entries)

        m_4.to_corner(UL)
        self.play(FadeIn(m_4))

        self.wait(2)
        self.play(self.plane.animate.apply_matrix(m_4_entries), i.animate.apply_matrix(m_4_entries),
                  j.animate.apply_matrix(m_4_entries), run_time=2)


class MultiplyingMatricesScene(Scene):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.background_plane_kwargs = {
            "color": GREY,
            "axis_config": {
                "color": GREY,
            },
            "background_line_style": {
                "stroke_color": GREY,
                "stroke_width": 1,
            },
        }
        self.background_plane = NumberPlane(**self.background_plane_kwargs)
        self.add(self.background_plane)
        self.plane = NumberPlane(x_range=[-10, 10, 1], y_range=[-10, 10, 1])
        self.add(self.plane)

        self.i = Vector([1, 0], color=GREEN_D, max_tip_length_to_length_ratio=0)
        self.j = Vector([0, 1], color=RED, max_tip_length_to_length_ratio=0)

        i_text = MathTex(r"\hat{i}", color=GREEN_D)
        i_bg = BackgroundRectangle(i_text, color=BLACK, fill_opacity=0.5, buff=0.1).move_to(i_text.center())
        i_label = VGroup(i_bg, i_text).add_updater(lambda l: l.next_to(self.i.get_end(), RIGHT))

        j_text = MathTex(r"\hat{j}", color=RED)
        j_bg = BackgroundRectangle(j_text, color=BLACK, fill_opacity=0.5, buff=0.1).move_to(j_text.center())
        j_label = VGroup(j_bg, j_text).add_updater(lambda l: l.next_to(self.j.get_end(), UP))

        self.add(self.i, self.j, i_label, j_label)

    def apply_matrix(self, matrix):
        self.play(self.plane.animate.apply_matrix(matrix), self.i.animate.apply_matrix(matrix),
                  self.j.animate.apply_matrix(matrix), run_time=2)
        self.wait(1)

    def construct(self):
        m_1 = np.array([[2, 0],
                        [0, 3]])
        m_2 = np.array([[0, -1],
                        [1, 0]])

        g_1 = create_matrix_with_box(m_1).to_corner(UL)
        g_2 = create_matrix_with_box(m_2).to_corner(UL)

        self.play(FadeIn(g_1))
        self.apply_matrix(m_1)
        self.remove(g_1)

        self.play(FadeIn(g_2))
        self.apply_matrix(m_2)
        self.remove(g_2)

        self.wait()

        m_3 = [[0, -3],
               [2, 0]]
        g_3 = create_matrix_with_box(m_3).to_corner(UL)

        self.play(FadeIn(g_3))

        self.wait(2)

        self.remove(g_3)

        multiplication = MathTex(
            r"\begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}",
            r"\begin{bmatrix} 2 & 0 \\ 0 & 3 \end{bmatrix}",
            r"=",
            r"\begin{bmatrix} 0 & -3 \\ 2 & 0 \end{bmatrix}"
        ).scale(2)
        bg = BackgroundRectangle(multiplication, color=BLACK, fill_opacity=0.5, buff=0.2).move_to(
            multiplication.center())
        self.play(FadeIn(bg), FadeIn(multiplication))

        self.wait()


class MatrixInverseScene(Scene):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.background_plane_kwargs = {
            "color": GREY,
            "axis_config": {
                "color": GREY,
            },
            "background_line_style": {
                "stroke_color": GREY,
                "stroke_width": 1,
            },
        }
        self.background_plane = NumberPlane(**self.background_plane_kwargs)
        self.add(self.background_plane)
        self.plane = NumberPlane(x_range=[-10, 10, 1], y_range=[-10, 10, 1])
        self.add(self.plane)

        self.i = Vector([1, 0], color=GREEN_D, max_tip_length_to_length_ratio=0)
        self.j = Vector([0, 1], color=RED, max_tip_length_to_length_ratio=0)

        i_text = MathTex(r"\hat{i}", color=GREEN_D)
        i_bg = BackgroundRectangle(i_text, color=BLACK, fill_opacity=0.5, buff=0.1).move_to(i_text.center())
        i_label = VGroup(i_bg, i_text).add_updater(lambda l: l.next_to(self.i.get_end(), RIGHT))

        j_text = MathTex(r"\hat{j}", color=RED)
        j_bg = BackgroundRectangle(j_text, color=BLACK, fill_opacity=0.5, buff=0.1).move_to(j_text.center())
        j_label = VGroup(j_bg, j_text).add_updater(lambda l: l.next_to(self.j.get_end(), UP))

        self.add(self.i, self.j, i_label, j_label)

    def apply_matrix(self, matrix):
        self.play(self.plane.animate.apply_matrix(matrix), self.i.animate.apply_matrix(matrix),
                  self.j.animate.apply_matrix(matrix), run_time=3)
        self.wait(1)

    def construct(self):
        m = [[1, 3],
             [2, 1]]
        g = create_matrix_with_box(m).to_corner(UL)

        self.play(FadeIn(g))
        self.apply_matrix(m)
        self.wait(1)
        self.remove(g)

        i = [[-1 / 5, 3 / 5],
             [2 / 5, -1 / 5]]
        g_2 = create_matrix_with_box(i).to_corner(UL)

        self.play(FadeIn(g_2))
        self.apply_matrix(i)
        self.wait(1)
        self.remove(g_2)

        self.wait()

        multiplication = MathTex(
            r"\begin{bmatrix} -\frac{1}{5} & \frac{3}{5} \\ \frac{2}{5} & -\frac{1}{5} \end{bmatrix}",
            r"\begin{bmatrix} 1 & 3 \\ 2 & 1 \end{bmatrix}",
            r"=",
            r"\begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}"
        ).scale(2)
        bg = BackgroundRectangle(multiplication, color=BLACK, fill_opacity=0.5, buff=0.2).move_to(
            multiplication.center())
        self.play(FadeIn(bg), FadeIn(multiplication))


class MatrixEigenValueAndEigenVectorIntroductionScene(Scene):
    def construct(self):
        title = Tex("Eigenvalues and Eigenvectors")

        self.play(Write(title))

        formula = MathTex(r"\text{det}(A - \lambda I) = 0")

        self.play(ReplacementTransform(title, formula), run_time=2)
        self.wait(2)

        self.play(FadeOut(formula))


class MatrixEigenValueAndEigenVectorScene(VectorScene):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.background_plane_kwargs = {
            "color": GREY,
            "axis_config": {
                "color": GREY,
            },
            "background_line_style": {
                "stroke_color": GREY,
                "stroke_width": 1,
            },
        }
        self.background_plane = NumberPlane(**self.background_plane_kwargs)
        self.add(self.background_plane)
        self.plane = NumberPlane(x_range=[-10, 10, 1], y_range=[-10, 10, 1])
        self.add(self.plane)

        self.i = Vector([1, 0], color=GREEN_D, max_tip_length_to_length_ratio=0)
        self.j = Vector([0, 1], color=RED, max_tip_length_to_length_ratio=0)

        i_text = MathTex(r"\hat{i}", color=GREEN_D)
        i_bg = BackgroundRectangle(i_text, color=BLACK, fill_opacity=0.5, buff=0.1).move_to(i_text.center())
        i_label = VGroup(i_bg, i_text).add_updater(lambda l: l.next_to(self.i.get_end(), RIGHT))

        j_text = MathTex(r"\hat{j}", color=RED)
        j_bg = BackgroundRectangle(j_text, color=BLACK, fill_opacity=0.5, buff=0.1).move_to(j_text.center())
        j_label = VGroup(j_bg, j_text).add_updater(lambda l: l.next_to(self.j.get_end(), UP))

        self.add(self.i, self.j, i_label, j_label)

    def apply_matrix(self, matrix, added_anim: list = None):
        if added_anim is None:
            added_anim = []

        self.play(
            self.plane.animate.apply_matrix(matrix),
            self.i.animate.apply_matrix(matrix),
            self.j.animate.apply_matrix(matrix),
            *added_anim,
            run_time=2
        )
        self.wait(1)

    def construct(self):
        m = [[3, 1],
             [0, 2]]
        inv = [[1/3, -1/6],
               [0, 1/2]]
        g = create_matrix_with_box(m).to_edge(UP).shift(2 * LEFT)

        self.play(FadeIn(g))

        vector_coordinates = [[-1, 1],
                              [1, 1],
                              [-1, -1],
                              [1, 1/2],
                              [1/2, 1],
                              [-1/2, 1],
                              [-1/2, -1],
                              [-1, 1/2]]
        vectors = []
        for vc in vector_coordinates:
            vectors.append(Vector(vc, color=YELLOW_A, max_tip_length_to_length_ratio=0))

        self.play(*[Create(v) for v in vectors], run_time=3)

        self.apply_matrix(m, added_anim=[v.animate.apply_matrix(m) for v in vectors])
        self.apply_matrix(inv, added_anim=[v.animate.apply_matrix(inv) for v in vectors])

        self.remove(*vectors[2:])

        self.play(vectors[0].animate.set_color(BLUE), vectors[1].animate.set_color(LIGHT_BROWN), run_time=1)

        self.apply_matrix(m, added_anim=[v.animate.apply_matrix(m) for v in vectors[:2]])


class ThatOneSpecificVectorScene(Scene):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.background_plane_kwargs = {
            "color": GREY,
            "axis_config": {
                "color": GREY,
            },
            "background_line_style": {
                "stroke_color": GREY,
                "stroke_width": 1,
            },
        }
        self.background_plane = NumberPlane(**self.background_plane_kwargs)
        self.add(self.background_plane)
        self.plane = NumberPlane(x_range=[-10, 10, 1], y_range=[-10, 10, 1])
        self.add(self.plane)

        self.i = Vector([1, 0], color=GREEN_D, max_tip_length_to_length_ratio=0)
        self.j = Vector([0, 1], color=RED, max_tip_length_to_length_ratio=0)

        i_text = MathTex(r"\hat{i}", color=GREEN_D)
        i_bg = BackgroundRectangle(i_text, color=BLACK, fill_opacity=0.5, buff=0.1).move_to(i_text.center())
        i_label = VGroup(i_bg, i_text).add_updater(lambda l: l.next_to(self.i.get_end(), RIGHT))

        j_text = MathTex(r"\hat{j}", color=RED)
        j_bg = BackgroundRectangle(j_text, color=BLACK, fill_opacity=0.5, buff=0.1).move_to(j_text.center())
        j_label = VGroup(j_bg, j_text).add_updater(lambda l: l.next_to(self.j.get_end(), UP))

        self.add(self.i, self.j, i_label, j_label)

    def apply_matrix(self, matrix, added_anim: list = None):
        if added_anim is None:
            added_anim = []

        self.play(
            self.plane.animate.apply_matrix(matrix),
            self.i.animate.apply_matrix(matrix),
            self.j.animate.apply_matrix(matrix),
            *added_anim,
            run_time=2
        )
        self.wait(1)

    def construct(self):
        m = [[3, 1],
             [0, 2]]
        inv = [[1/3, -1/6],
               [0, 1/2]]
        g = create_matrix_with_box(m).to_edge(UP).shift(2 * LEFT)

        self.add(g)

        base_vec_coordinates = [-0.5, 0.5]
        vecs = [(Vector([coff * base_vec_coordinates[0], coff * base_vec_coordinates[1]], color=YELLOW),
                 Vector([2 * coff * base_vec_coordinates[0], 2 * coff * base_vec_coordinates[1]], color=YELLOW))
                for coff in range(-7, 8)]

        self.play([Create(v[0]) for v in vecs])
        self.apply_matrix(m, added_anim=[ReplacementTransform(v[0], v[1]) for v in vecs])
        self.remove(*[v[1] for v in vecs])
        self.apply_matrix(inv)
        self.wait()

        u = Vector([-1, 1], color=YELLOW, max_tip_length_to_length_ratio=0)
        u_text = MathTex(r"\hat{u}", color=YELLOW)
        u_bg = BackgroundRectangle(u_text, color=BLACK, fill_opacity=0.5, buff=0.1).move_to(u_text.center())
        u_label = VGroup(u_bg, u_text).add_updater(lambda l: l.next_to(u.get_end(), RIGHT))
        self.play(Create(u), Write(u_label))

        self.apply_matrix(m, added_anim=[u.animate.apply_matrix(m)])

        eigenvalues = create_math_text_with_background(r"\lambda_1 = 2 \\ \lambda_2 = 3").to_edge(DOWN).shift(2 * RIGHT)
        self.play(Write(eigenvalues))


class CanYouSpotTheEigenvectorsScene(Scene):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.background_plane_kwargs = {
            "color": GREY,
            "axis_config": {
                "color": GREY,
            },
            "background_line_style": {
                "stroke_color": GREY,
                "stroke_width": 1,
            },
        }
        self.background_plane = NumberPlane(**self.background_plane_kwargs)
        self.add(self.background_plane)
        self.plane = NumberPlane(x_range=[-10, 10, 1], y_range=[-10, 10, 1])
        self.add(self.plane)

        self.i = Vector([1, 0], color=GREEN_D, max_tip_length_to_length_ratio=0)
        self.j = Vector([0, 1], color=RED, max_tip_length_to_length_ratio=0)

        i_text = MathTex(r"\hat{i}", color=GREEN_D)
        i_bg = BackgroundRectangle(i_text, color=BLACK, fill_opacity=0.5, buff=0.1).move_to(i_text.center())
        i_label = VGroup(i_bg, i_text).add_updater(lambda l: l.next_to(self.i.get_end(), RIGHT))

        j_text = MathTex(r"\hat{j}", color=RED)
        j_bg = BackgroundRectangle(j_text, color=BLACK, fill_opacity=0.5, buff=0.1).move_to(j_text.center())
        j_label = VGroup(j_bg, j_text).add_updater(lambda l: l.next_to(self.j.get_end(), UP))

        self.add(self.i, self.j, i_label, j_label)

    def apply_matrix(self, matrix, added_anim: list = None):
        if added_anim is None:
            added_anim = []

        self.play(
            self.plane.animate.apply_matrix(matrix),
            self.i.animate.apply_matrix(matrix),
            self.j.animate.apply_matrix(matrix),
            *added_anim,
            run_time=2
        )
        self.wait(1)

    def construct(self):
        m = [[2, 1],
             [1, 2]]
        inv = [[2/3, -1/3],
               [-1/3, 2/3]]
        g = create_matrix_with_box(m).to_corner(UL)
        sqrt_2 = math.sqrt(2)

        self.play(Write(g))

        self.apply_matrix(m)
        self.wait()
        self.apply_matrix(inv)

        unit_square = Polygon(
            self.plane.c2p(0, 0),  # Bottom-left corner
            self.plane.c2p(1, 0),  # Bottom-right corner
            self.plane.c2p(1, 1),  # Top-right corner
            self.plane.c2p(0, 1),  # Top-left corner
            color=YELLOW,
            fill_color=YELLOW,
            fill_opacity=0.5
        )

        self.play(Create(unit_square))

        self.apply_matrix(m, added_anim=[unit_square.animate.apply_matrix(m)])
        self.wait()
        self.apply_matrix(inv, added_anim=[unit_square.animate.apply_matrix(inv)])
        self.wait()
        self.remove(unit_square)
        self.wait()

        u = Vector([sqrt_2/2, sqrt_2/2], color=YELLOW, max_tip_length_to_length_ratio=0)
        u_text = MathTex(r"\hat{u}", color=YELLOW)
        u_bg = BackgroundRectangle(u_text, color=BLACK, fill_opacity=0.5, buff=0.1).move_to(u_text.center())
        u_label = VGroup(u_bg, u_text).add_updater(lambda l: l.next_to(u.get_end(), RIGHT))
        self.play(Create(u), Write(u_label))

        self.apply_matrix(m, added_anim=[u.animate.apply_matrix(m)])
        self.wait()
        self.apply_matrix(inv, added_anim=[u.animate.apply_matrix(inv)])
        self.wait()

        v = Vector([-sqrt_2 / 2, sqrt_2 / 2], color=BLUE_D, max_tip_length_to_length_ratio=0)
        v_text = MathTex(r"\hat{v}", color=BLUE_D)
        v_bg = BackgroundRectangle(v_text, color=BLACK, fill_opacity=0.5, buff=0.1).move_to(v_text.center())
        v_label = VGroup(v_bg, v_text).add_updater(lambda l: l.next_to(v.get_end(), RIGHT))
        self.play(Create(v), Write(v_label))

        self.apply_matrix(m, added_anim=[u.animate.apply_matrix(m), v.animate.apply_matrix(m)])
        self.wait()
        self.apply_matrix(inv, added_anim=[u.animate.apply_matrix(inv), v.animate.apply_matrix(m)])
        self.wait()

        r = [[sqrt_2 / 2, sqrt_2 / 2],
             [-sqrt_2 / 2, sqrt_2 / 2]]
        r_matrix = MathTex(
            r"\begin{bmatrix}"
            r"\frac{\sqrt{2}}{2} & -\frac{\sqrt{2}}{2} \\"
            r"\frac{\sqrt{2}}{2} & \frac{\sqrt{2}}{2}"
            r"\end{bmatrix}"
        )
        r_box = Rectangle(
            width=r_matrix.width + 0.5,
            height=r_matrix.height + 0.5,
            color=BLACK,
            fill_opacity=0.5,
            stroke_width=0
        ).move_to(r_matrix.get_center())
        r_gp = VGroup(r_box, r_matrix)

        self.play(Write(r_gp))
        self.play(r_gp.animate.to_corner(DR))

        self.apply_matrix(r, added_anim=[u.animate.apply_matrix(r), v.animate.apply_matrix(r)])

        s = [[3, 0],
             [0, 1]]
        s_matrix = MathTex(
            r"\begin{bmatrix}"
            r"3 & 0 \\"
            r"0 & 1"
            r"\end{bmatrix}"
        )
        s_box = Rectangle(
            width=s_matrix.width + 0.5,
            height=s_matrix.height + 0.5,
            color=BLACK,
            fill_opacity=0.5,
            stroke_width=0
        ).move_to(s_matrix.get_center())
        s_gp = VGroup(s_box, s_matrix)

        self.play(Write(s_gp))
        self.play(s_gp.animate.next_to(r_gp, LEFT))

        self.apply_matrix(s, added_anim=[u.animate.apply_matrix(s), v.animate.apply_matrix(s)])

        r_inv = [[sqrt_2 / 2, -sqrt_2 / 2],
                 [sqrt_2 / 2, sqrt_2 / 2]]
        r_inv_matrix = MathTex(
            r"\begin{bmatrix}"
            r"\frac{\sqrt{2}}{2} & -\frac{\sqrt{2}}{2} \\"
            r"\frac{\sqrt{2}}{2} & \frac{\sqrt{2}}{2}"
            r"\end{bmatrix}^{-1}"
        )
        r_inv_box = Rectangle(
            width=r_inv_matrix.width + 0.5,
            height=r_inv_matrix.height + 0.5,
            color=BLACK,
            fill_opacity=0.5,
            stroke_width=0
        ).move_to(r_inv_matrix.get_center())
        r_inv_gp = VGroup(r_inv_box, r_inv_matrix)

        self.play(Write(r_inv_gp))
        self.play(r_inv_gp.animate.next_to(s_gp, LEFT))

        self.apply_matrix(r_inv, added_anim=[u.animate.apply_matrix(r_inv), v.animate.apply_matrix(r_inv)])

        r_tran_matrix = MathTex(
            r"\begin{bmatrix}"
            r"\frac{\sqrt{2}}{2} & -\frac{\sqrt{2}}{2} \\"
            r"\frac{\sqrt{2}}{2} & \frac{\sqrt{2}}{2}"
            r"\end{bmatrix}^T"
        )
        r_tran_box = Rectangle(
            width=r_tran_matrix.width + 0.5,
            height=r_tran_matrix.height + 0.5,
            color=BLACK,
            fill_opacity=0.5,
            stroke_width=0
        ).move_to(r_tran_matrix.get_center())
        r_tran_gp = VGroup(r_tran_box, r_tran_matrix).next_to(s_gp, LEFT)

        r_inv_gp.become(r_tran_gp)

        self.play(FadeIn(r_inv_gp))

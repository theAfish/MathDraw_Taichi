import taichi as ti

ti.init()

v_res = 500

@ti.data_oriented
class Rasterizer:
    def __init__(self):
        self.triangles = []
        self.camera = Camera()
        self.res = self.camera.res
        self.M_mvp = ti.Matrix.field(4, 4, dtype=ti.f32, shape=())
        self.window = ti.ui.Window("testing", res=self.res)
        self.canvas = self.window.get_canvas()
        self.image = ti.Vector.field(3, dtype=ti.f32, shape=self.res)

        self.get_mvp_mat()

    def rotate_z_mat(self, angle):
        return ti.Matrix([[ti.cos(angle), -ti.sin(angle), 0.0, 0.0],
                          [ti.sin(angle), ti.cos(angle), 0.0, 0.0],
                          [0.0, 0.0, 1.0, 0.0],
                          [0.0, 0.0, 0.0, 1.0]])

    @ti.kernel
    def get_mvp_mat(self):
        self.M_mvp[None] = self.camera.get_vp_mat() @ self.camera.get_pers_mat() @ self.camera.get_cam_mat()

    def draw_line(self, begin, end):
        line_color = [1., 1., 1.]
        x1 = begin.x
        y1 = begin.y
        x2 = end.x
        y2 = end.y
        xe, ye, x, y = 0, 0, 0, 0
        dx = int(x2 - x1)
        dy = int(y2 - y1)
        dx1 = ti.abs(dx)
        dy1 = ti.abs(dy)
        px = 2 * dy1 - dx1
        py = 2 * dx1 - dy1
        if dy1 <= dx1:
            if dx >= 0:
                x = x1
                y = y1
                xe = x2
            else:
                x = x2
                y = y2
                xe = x1

    @ti.kernel
    def draw_triangle(self, triangle: ti.template()):
        pa = ti.Vector([triangle.vertex[0].x, triangle.vertex[0].y])
        pb = ti.Vector([triangle.vertex[1].x, triangle.vertex[1].y])
        pc = ti.Vector([triangle.vertex[2].x, triangle.vertex[2].y])
        min_pos = ti.cast(ti.min(ti.min(pa, pb), pc) - 0.5, ti.i32)
        min_pos = ti.max(min_pos, ti.Vector([0, 0]))
        max_pos = ti.cast(ti.max(ti.max(pa, pb), pc) + 0.5, ti.i32)
        max_pos = ti.min(max_pos, self.res)
        for x, y in ti.ndrange((min_pos.x, max_pos.x), (min_pos.y, max_pos.y)):
            if triangle.inside(ti.Vector([x + 0.5, y + 0.5])):
                self.image[x, y] = triangle.color[None]

    def draw(self):
        for i in range(len(self.triangles)):
            new_triangle = Triangle(color=self.triangles[i].color[None])
            for v in range(3):
                p = self.M_mvp[None] @ self.triangles[i].vertex[v]
                new_triangle.set_vertex(v, p)
            self.draw_triangle(new_triangle)

    def add_triangle(self, triangle):
        self.triangles.append(triangle)

    def show(self):
        t = 0
        while self.window.running:
            self.image.fill(0)
            self.draw()
            self.canvas.set_image(self.image)
            self.window.show()
            self.camera.debug_move(t)
            self.get_mvp_mat()
            t += 1


@ti.data_oriented
class Camera:
    def __init__(self, fov=60, aspect_ratio=16 / 9, v_res=500):
        self.res = (int(v_res * aspect_ratio), v_res)
        self.fov = fov
        self.aspect_ratio = aspect_ratio
        self.z_near = 0.1
        self.z_far = 100.0
        self.vup = ti.Vector([0.0, 1.0, 0.0])
        self.look_from = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.look_at = ti.Vector.field(3, dtype=ti.f32, shape=())

        self.reset()

    def reset(self):
        self.look_from[None] = [2.0, 0.0, 0.0]
        self.look_at[None] = [0.0, 0.0, 2.0]

    def debug_move(self, t):
        self.look_at[None] = [0.0, 0.0, ti.sin(t * 0.1) * 2]

    @ti.func
    def get_cam_mat(self):
        w = (self.look_from[None] - self.look_at[None]).normalized()
        u = (self.vup.cross(w)).normalized()
        v = w.cross(u)
        M_cam = ti.Matrix(
            [[u.x, u.y, u.z, 0.],
             [v.x, v.y, v.z, 0.],
             [w.x, w.y, w.z, 0.],
             [0.0, 0.0, 0.0, 1.]]) @ ti.Matrix(
            [[1., 0., 0., -self.look_from[None].x],
             [0., 1., 0., -self.look_from[None].y],
             [0., 0., 1., -self.look_from[None].z],
             [0., 0., 0., 1.]])
        return M_cam

    @ti.func
    def get_pers_mat(self):
        t = self.z_near * ti.tan(self.fov / 2)
        r = t * self.res[0] / self.res[1]
        n = -self.z_near
        f = -self.z_far
        M_pers = ti.Matrix([[n, 0., 0., 0.],
                            [0., n, 0., 0.],
                            [0., 0., n+f, -n*f],
                            [0., 0., 1., 0.]])
        M_orth = ti.Matrix([[1 / r, 0., 0., 0.],
                           [0., 1 / t, 0., 0.],
                           [0., 0., 2 / (n - f), - (n + f) / (n - f)],
                           [0., 0., 0., 1.]])
        return M_pers @ M_orth

    @ti.func
    def get_vp_mat(self):
        nx = self.res[0]
        ny = self.res[1]
        return ti.Matrix([[nx / 2, 0., 0., (nx - 1) / 2],
                          [0., ny / 2, 0., (ny - 1) / 2],
                          [0., 0., 1., 0.],
                          [0., 0., 0., 1.]])


@ti.data_oriented
class Triangle:
    def __init__(self, color=None):
        if color is None:
            color = [1.0, 1.0, 1.0]
        self.vertex = ti.Vector.field(4, dtype=ti.f32, shape=3)
        self.color = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.color[None] = color

    def set_vertex(self, i, point):
        self.vertex[i] = point / point[3]

    @ti.func
    def inside(self, pos):
        is_inside = False
        pa = ti.Vector([self.vertex[0].x, self.vertex[0].y])
        pb = ti.Vector([self.vertex[1].x, self.vertex[1].y])
        pc = ti.Vector([self.vertex[2].x, self.vertex[2].y])
        a = (pb - pa).cross(pos - pa)
        b = (pc - pb).cross(pos - pb)
        c = (pa - pc).cross(pos - pc)
        if a * b >= 0 and a * c >= 0:
            is_inside = True
        return is_inside


p1 = ti.Vector([0.0, 0.0, -2.0, 1])
p2 = ti.Vector([0.0, 0.0, 2.0, 1])
p3 = ti.Vector([0.0, 2.0, 0.0, 1])
tri = Triangle()
tri.set_vertex(0, p1)
tri.set_vertex(1, p2)
tri.set_vertex(2, p3)

point = ti.Vector([1.1, 5.0, 5.0, 0.0])
tri2 = Triangle(color=[0.0, 1.0, 0.5])
tri2.set_vertex(0, p1 + point)
tri2.set_vertex(1, p2 + point)
tri2.set_vertex(2, p3 + point)

rst = Rasterizer()
rst.add_triangle(tri)
rst.add_triangle(tri2)

rst.show()

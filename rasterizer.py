import taichi as ti

ti.init(arch=ti.gpu)
# ti.init()

v_res = 500

RED = [1.0, 0.0, 0.0]
GREEN = [0.0, 1.0, 0.0]
BLUE = [0.0, 0.0, 1.0]
CYAN_GREEN = [0.0, 1.0, 0.5]

@ti.data_oriented
class Rasterizer:
    def __init__(self):
        self.triangles = None
        self.models = []
        self.camera = Camera()
        self.res = (int(self.camera.res[0]/2), int(self.camera.res[1]/2))
        self.cam_res = self.camera.res
        self.M_mvp = ti.Matrix.field(4, 4, dtype=ti.f32, shape=())
        self.window = ti.ui.Window("testing", res=self.res)
        self.canvas = self.window.get_canvas()
        self.pre_image = ti.Vector.field(3, dtype=ti.f32, shape=self.cam_res)
        self.depth_map = ti.field(dtype=ti.f32, shape=self.cam_res)
        self.image = ti.Vector.field(3, dtype=ti.f32, shape=self.res)

        self.get_mvp_mat()

    def add_model(self, model):
        self.models.append(model)

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

    def draw_models(self):
        for i in ti.static(range(len(self.models))):
            self.draw_model(i)

    @ti.kernel
    def draw_model(self, m:ti.template()):
        for f in range(self.models[m].face_num[None]):
            p1, p2, p3 = self.models[m].get_vertex(f)
            p1 = self.M_mvp[None] @ p1
            p2 = self.M_mvp[None] @ p2
            p3 = self.M_mvp[None] @ p3
            pa = ti.Vector([p1.x, p1.y])/p1[3]
            pb = ti.Vector([p2.x, p2.y])/p2[3]
            pc = ti.Vector([p3.x, p3.y])/p3[3]
            min_pos = ti.cast(ti.min(ti.min(pa, pb), pc) - 0.5, ti.i32)
            min_pos = ti.max(min_pos, ti.Vector([0, 0]))
            max_pos = ti.cast(ti.max(ti.max(pa, pb), pc) + 0.5, ti.i32)
            max_pos = ti.min(max_pos, self.cam_res)
            for x, y in ti.ndrange((min_pos.x, max_pos.x), (min_pos.y, max_pos.y)):
                p = ti.Vector([x + 0.5, y + 0.5])
                if self.models[m].inside(p, pa, pb, pc):
                    j = ((p.y-pc.y)*(pa.x-pc.x)-(p.x-pc.x)*(pa.y-pc.y))/((pb.y-pc.y)*(pa.x-pc.x)-(pb.x-pc.x)*(pa.y-pc.y))
                    i = ((p.y-pb.y)*(pc.x-pb.x)-(p.x-pb.x)*(pc.y-pb.y))/((pa.y-pb.y)*(pc.x-pb.x)-(pa.x-pb.x)*(pc.y-pb.y))
                    k = 1.0 - i - j
                    z = p1[0] * i + p2[0] * j + p3[0] * k
                    if ti.atomic_min(z, self.depth_map[x, y]) <= z: #z <= self.depth_map[x, y]:
                        self.depth_map[x, y] = z
                        n = self.models[m].get_normal(f)
                        pos_xy = (self.vec4to3(p1) * i + self.vec4to3(p2) * j + self.vec4to3(p3) * k).normalized()
                        self.pre_image[x, y] = self.models[m].color[None] * ti.abs(n.dot(pos_xy))

    @ti.func
    def vec4to3(self, vec4):
        return ti.Vector([vec4[0], vec4[1], vec4[2]])

    @ti.kernel
    def draw_triangles(self):
        for t in range(self.triangles.triangle_num[None]):
            p1 = self.M_mvp[None] @ self.triangles.A[t]
            p2 = self.M_mvp[None] @ self.triangles.B[t]
            p3 = self.M_mvp[None] @ self.triangles.C[t]
            pa = ti.Vector([p1.x, p1.y])/p1[3]
            pb = ti.Vector([p2.x, p2.y])/p2[3]
            pc = ti.Vector([p3.x, p3.y])/p3[3]
            min_pos = ti.cast(ti.min(ti.min(pa, pb), pc) - 0.5, ti.i32)
            min_pos = ti.max(min_pos, ti.Vector([0, 0]))
            max_pos = ti.cast(ti.max(ti.max(pa, pb), pc) + 0.5, ti.i32)
            max_pos = ti.min(max_pos, self.cam_res)
            for x, y in ti.ndrange((min_pos.x, max_pos.x), (min_pos.y, max_pos.y)):
                p = ti.Vector([x + 0.5, y + 0.5])
                if self.triangles.inside(p, pa, pb, pc):
                    j = ((p.y-pc.y)*(pa.x-pc.x)-(p.x-pc.x)*(pa.y-pc.y))/((pb.y-pc.y)*(pa.x-pc.x)-(pb.x-pc.x)*(pa.y-pc.y))
                    i = ((p.y-pb.y)*(pc.x-pb.x)-(p.x-pb.x)*(pc.y-pb.y))/((pa.y-pb.y)*(pc.x-pb.x)-(pa.x-pb.x)*(pc.y-pb.y))
                    k = 1.0 - i - j
                    z = p1[0] * i + p2[0] * j + p3[0] * k
                    if ti.atomic_min(z, self.depth_map[x, y]) <= z: #z <= self.depth_map[x, y]:
                        self.depth_map[x, y] = z
                        self.pre_image[x, y] = self.triangles.color[t]

    @ti.kernel
    def anti_aliasing(self):
        for i, j in self.image:
            self.image[i, j] = (self.pre_image[2*i, 2*j]
                                + self.pre_image[2*i, 2*j+1]
                                + self.pre_image[2*i+1, 2*j]
                                + self.pre_image[2*i+1, 2*j+1]) / 4

    def set_triangles(self, triangles):
        self.triangles = triangles

    def show(self):
        t = 0
        while self.window.running:
            self.depth_map.fill(1e5)
            self.pre_image.fill(0)
            # self.draw_triangles()
            self.draw_models()
            self.anti_aliasing()
            self.canvas.set_image(self.image)
            self.window.show()
            self.camera.debug_move(t)
            self.get_mvp_mat()
            t += 1


@ti.data_oriented
class Camera:
    def __init__(self, fov=60, aspect_ratio=16 / 9, v_res=500):
        self.res = (int(v_res * aspect_ratio*2), v_res*2)
        self.fov = fov
        self.aspect_ratio = aspect_ratio
        self.z_near = 0.001
        self.z_far = 100.0
        self.vup = ti.Vector([0.0, 1.0, 0.0])
        self.look_from = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.look_at = ti.Vector.field(3, dtype=ti.f32, shape=())

        self.reset()

    def reset(self):
        self.look_from[None] = [10.0, 10.0, -5.0]
        self.look_at[None] = [0.0, 0.0, 0.0]

    def debug_move(self, t):
        # self.look_at[None] = [0.0, 0.0, ti.sin(t * 0.01) * 2]
        r = 5.0
        omega = -0.001
        self.look_from[None] = [-r*ti.cos(t * omega), 0.0, -r*ti.sin(t * omega)]
        # theta = 3.14/4 * 3
        # self.look_from[None] = [-r*ti.cos(theta), 5.0, -r*ti.sin(theta)]


    @ti.func
    def get_cam_mat(self):
        w = -(self.look_from[None] - self.look_at[None]).normalized()
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
class Triangles:
    def __init__(self, max_num=1024):
        self.n = max_num

        self.A = ti.Vector.field(4, dtype=ti.f32)
        self.B = ti.Vector.field(4, dtype=ti.f32)
        self.C = ti.Vector.field(4, dtype=ti.f32)
        self.vertices = ti.root.dense(ti.i, self.n).place(self.A, self.B, self.C)
        self.triangle_num = ti.field(dtype=ti.i32, shape=())
        self.color = ti.Vector.field(3, dtype=ti.f32, shape=max_num)

    def reset(self):
        self.triangle_num[None] = 0

    def add_triangle(self, p1, p2, p3, color=None):
        if color is None:
            color = [1.0, 1.0, 1.0]
        i = self.triangle_num[None]
        self.A[i] = [p1.x, p1.y, p1.z, 1.0]
        self.B[i] = [p2.x, p2.y, p2.z, 1.0]
        self.C[i] = [p3.x, p3.y, p3.z, 1.0]
        self.color[i] = color
        self.triangle_num[None] += 1

    @staticmethod
    @ti.func
    def inside(pos, pa, pb, pc):
        is_inside = False
        a = (pb - pa).cross(pos - pa)
        b = (pc - pb).cross(pos - pb)
        c = (pa - pc).cross(pos - pc)
        if a * b >= 0 and a * c >= 0:
            is_inside = True
        return is_inside


import taichi as ti
from vispy import io


@ti.data_oriented
class Model:
    def __init__(self, max_vertex_num=25000, max_faces_num=50000):
        self.vertex_num = ti.field(dtype=ti.i32, shape=())
        self.vertex_num[None] = 0
        self.face_num = ti.field(dtype=ti.i32, shape=())
        self.face_num[None] = 0
        self.vertex = ti.Vector.field(4, dtype=ti.f32, shape=max_vertex_num)
        self.vertex_normal = ti.Vector.field(3, dtype=ti.f32, shape=max_vertex_num)
        self.face = ti.Vector.field(3, dtype=ti.i32, shape=max_faces_num)
        self.color = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.color[None] = [1.0, 1.0, 1.0]

    def clear_face(self):
        self.vertex_num[None] = 0
        self.face_num[None] = 0

    def from_obj(self, filename):
        # this part and the obj models are copied from https://github.com/MicroappleMA/path_tracing_obj;
        self.clear_face()
        vertex, face, normal, texture = io.read_mesh(filename)
        for index in range(len(vertex)):
            self.vertex[index] = ti.Vector([vertex[index][0], vertex[index][1], vertex[index][2], 1])
        for index in range(len(face)):
            self.face[index] = ti.Vector([face[index][0], face[index][1], face[index][2]])
        for index in range(len(normal)):
            self.vertex_normal[index] = ti.Vector([normal[index][0], normal[index][1], normal[index][2]])
        self.vertex_num[None] = len(vertex)
        self.face_num[None] = len(face)

    @ti.kernel
    def scale(self, ratio:ti.f32):
        s = ti.Matrix([[ratio, 0.0, 0.0, 0.0],
                       [0.0, ratio, 0.0, 0.0],
                       [0.0, 0.0, ratio, 0.0],
                       [0.0, 0.0, 0.0, 1.0]])
        for v in self.vertex:
            self.vertex[v] = s @ self.vertex[v]

    @ti.kernel
    def translate(self, x:ti.f32, y:ti.f32, z: ti.f32):
        t = ti.Matrix([[1.0, 0.0, 0.0, x],
                       [0.0, 1.0, 0.0, y],
                       [0.0, 0.0, 1.0, z],
                       [0.0, 0.0, 0.0, 1.0]])
        for v in self.vertex:
            self.vertex[v] = t @ self.vertex[v]

    @ti.func
    def get_normal(self, face_idx):
        Pab = self.vertex[self.face[face_idx][1]] - self.vertex[self.face[face_idx][0]]
        pab = ti.Vector([Pab[0], Pab[1], Pab[2]])
        Pac = self.vertex[self.face[face_idx][2]] - self.vertex[self.face[face_idx][0]]
        pac = ti.Vector([Pac[0], Pac[1], Pac[2]])
        return pab.cross(pac).normalized()

    @ti.func
    def get_vertex(self, face_idx):
        return self.vertex[self.face[face_idx][0]], self.vertex[self.face[face_idx][1]], self.vertex[self.face[face_idx][2]]

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

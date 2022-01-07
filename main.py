import taichi as ti
from rasterizer import *
from model import *

ti.init(arch=ti.gpu)

model = Model()
model.from_obj("./models/bunny.obj")
model.scale(20.0)
model.translate(0, -5.0, 0)

rst = Rasterizer()
rst.add_model(model)

rst.show()

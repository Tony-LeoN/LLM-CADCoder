import FreeCAD as App
import Part

doc = App.newDocument("MyModel")

# 创建一个立方体
box = Part.makeBox(10, 20, 30)
Part.show(box)

# 创建圆柱并做布尔运算
cylinder = Part.makeCylinder(5, 40)
cut = box.cut(cylinder)
Part.show(cut)

doc.recompute()
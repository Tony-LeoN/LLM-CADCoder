print("1. 成功进入脚本！准备导入 FreeCAD...")
import FreeCAD as App
import Part
print("2. 成功导入 FreeCAD 模块！准备开始建模...")

def create_machined_plate():
    # 几何生成逻辑与之前完全一致
    pts = [
        App.Vector(28, 20, 0), App.Vector(-28, 20, 0),
        App.Vector(-30, 18, 0), App.Vector(-30, -18, 0),
        App.Vector(-28, -20, 0), App.Vector(28, -20, 0),
        App.Vector(30, -18, 0), App.Vector(30, 18, 0),
        App.Vector(28, 20, 0)
    ]
    edges = [Part.makeLine(pts[i], pts[i+1]) for i in range(len(pts)-1)]
    wire = Part.Wire(edges)
    face = Part.Face(wire)
    plate = face.extrude(App.Vector(0, 0, 8))
    
    hole_tools = []
    # 正面孔
    for x in [-25, 25]:
        for y in [-15, 15]:
            thru = Part.makeCylinder(4.5/2.0, 10, App.Vector(x, y, -1))
            cbore = Part.makeCylinder(8.0/2.0, 5.6, App.Vector(x, y, 3.4))
            hole_tools.append(thru.fuse(cbore))
            
    # 反面孔
    for x in [-17.5, 17.5]:
        for y in [-5.5, 5.5]:
            thru = Part.makeCylinder(3.4/2.0, 10, App.Vector(x, y, -1))
            cbore = Part.makeCylinder(6.0/2.0, 4.0, App.Vector(x, y, -1))
            hole_tools.append(thru.fuse(cbore))

    final_part = plate
    for tool in hole_tools:
        final_part = final_part.cut(tool)
        
    return final_part

# ==========================================
# 核心执行区 (直接执行，不加 if __name__ 判断)
# ==========================================
print("3. 开始在后台构建三维模型...")

# 1. 创建虚拟文档
doc_name = "HeadlessDoc"
doc = App.newDocument(doc_name)

# 2. 运行建模逻辑
shape = create_machined_plate()
obj = doc.addObject("Part::Feature", "MachinedPlate")
obj.Shape = shape
doc.recompute()

# 3. 导出模型 (注意字符串前面加了 'r'，防止 \0 转义破坏路径！)
output_step = r"D:\Projects\Python\LLM-CADCoder\DataFlow\01.RawPDFWithSTEP\output_machined_plate.step"
print(f"4. 准备导出 STEP 文件至: {output_step}")

try:
    shape.exportStep(output_step)
    print("5. 建模与导出全部完成！顺利结束。")
except Exception as e:
    print(f"导出失败，发生错误: {e}")
import cadquery as cq

def build_part():
    # 1. 创建基础毛坯：长60, 宽40, 沿Z轴拉伸厚度为8
    # 默认建立在 XY 平面上，关于原点对称
    base_block = cq.Workplane("XY").box(60, 40, 8)
    
    # 2. 倒角操作：选择平行于Z轴的4条边，进行 C2 倒角
    part_with_chamfer = base_block.edges("|Z").chamfer(2)
    
    # 3. 建立外侧正面沉头孔
    # 选择顶面(>Z)建立工作平面，绘制 50x30 的构造矩形获取4个顶点
    part_front_holes = (
        part_with_chamfer.faces(">Z").workplane()
        .rect(50, 30, forConstruction=True).vertices()
        .cboreHole(diameter=4.5, cboreDiameter=8.0, cboreDepth=4.6)
    )
    
    # 4. 建立内侧反面沉头孔
    # 选择底面(<Z)建立工作平面，以应对图纸中的“反面沉孔”技术要求
    # 绘制 35x11 的构造矩形获取4个顶点
    final_part = (
        part_front_holes.faces("<Z").workplane()
        .rect(35, 11, forConstruction=True).vertices()
        .cboreHole(diameter=3.4, cboreDiameter=6.0, cboreDepth=3.0)
    )
    
    return final_part

# 生成零件实体
part = build_part()

# 如果你在 CQ-Editor 环境中运行，可以使用 show_object 进行可视化渲染
# show_object(part, options={"color": "lightgray", "alpha": 0})

# 如果需要导出为 STEP 格式进行后续几何分析或处理：
cq.exporters.export(part, "D:\datasets\零件图纸\图纸_tmp\X024-14-005-A_CQ.step")
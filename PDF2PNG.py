import os
import fitz  # 导入 PyMuPDF（fitz）
from tqdm import tqdm
import shutil

def find_pdfs(folder_path):
    pdf_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.pdf'):  # 忽略大小写
                pdf_files.append(os.path.join(root, file))
    return pdf_files

def is_multipage_pdf(pdf_path):
    try:
        # 打开 PDF 文件
        doc = fitz.open(pdf_path)
        # 获取 PDF 文件的页数
        num_pages = doc.page_count
        return num_pages > 1
    except Exception as e:
        # 捕获异常并打印出错文件的路径
        print(f"无法读取 PDF 文件: {pdf_path}, 错误: {e}")
        return False


def find_multipage_pdfs(folder_path):
    pdf_files = []
    multipage_pdfs = []  # 存储所有多页 PDF 文件的路径

    # 遍历文件夹中的所有文件，收集所有的 PDF 文件
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))

    # 使用 tqdm 显示进度条
    for pdf_path in tqdm(pdf_files, desc="Processing PDFs", unit="file"):
        if is_multipage_pdf(pdf_path):
            multipage_pdfs.append(pdf_path)  # 收集多页 PDF 文件路径

    return multipage_pdfs

# 缩放因子
ZOOM = 7

# 主函数
if __name__ == "__main__":
    # 示例用法
    folder_path = 'D:\\datasets\\零件图纸\\图纸-0610\\2D(PDF)'  # 替换为你的文件夹路径
    pdf_files = find_pdfs(folder_path)

    for pdf in tqdm(pdf_files, desc="生成PNG中", unit="file"):
        if '97.多页' not in pdf:
            directory = os.path.dirname(pdf)
            directory = os.path.dirname(directory)
            directory = os.path.dirname(directory)
            directory = os.path.join(directory, '95.PDF-生成-PNG', os.path.basename(os.path.dirname(pdf)))
            os.makedirs(directory, exist_ok=True)
            PDFdoc = fitz.open(pdf)
            folder_name = pdf.split("\\")[-1].split(".")[0]  # 按源文件名新建文件夹
            for pg in range(PDFdoc.page_count):  # 根据PDF的页数,按页提取图片
                page = PDFdoc[pg]
                # 增强图片分辨率
                zoom_x = ZOOM  # 设置每页的水平缩放因子
                zoom_y = ZOOM  # 设置每页的垂直缩放因子
                mat = fitz.Matrix(zoom_x, zoom_y)
                pix = page.get_pixmap(matrix=mat)
                # 按原PDF名称新建文件夹并按顺序保存图片
                # if not os.path.exists(IMG_PATH + folder_name):  # 判断文件夹是否已存在
                #     os.makedirs(IMG_PATH + folder_name)  # 不存在则新建，存在就跳过这行
                # pix.writeImage(IMG_PATH + folder_name + "\\{}.png".format(str(pg + 1)))  # 按PDF中的页面顺序命名并保存图片
                if PDFdoc.page_count == 1:
                    pix.save(directory + '/' + folder_name + ".png")  # 按PDF中的页面顺序命名并保存图片
                else:
                    pass

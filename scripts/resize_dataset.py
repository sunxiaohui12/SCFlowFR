import os
from PIL import Image
import concurrent.futures
import threading

# 设置输入和输出文件夹路径
input_folder = "load/FFHQ/FFHQ-images1024x1024"
output_folder = "load/FFHQ/FFHQ512"

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


# 定义单张图像处理函数
def process_image(filename):
    try:
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
            output_path = os.path.join(output_folder, filename)
            # 检查目标文件是否已存在
            if os.path.exists(output_path):
                # print(f"跳过: {filename} (已存在)")
                return
            img_path = os.path.join(input_folder, filename)
            img = Image.open(img_path)
            resized_img = img.resize((512, 512), Image.BICUBIC)
            resized_img.save(output_path)
            print(f"已处理: {filename} (线程: {threading.current_thread().name})")
    except Exception as e:
        print(f"处理 {filename} 时出错: {str(e)}")


# 获取所有图像文件
image_files = [
    f
    for f in os.listdir(input_folder)
    if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))
]

# 使用线程池并行处理
print(os.cpu_count())
max_workers = os.cpu_count() or 4  # 使用 CPU 核心数作为线程数，默认为 4
with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    executor.map(process_image, image_files)

print("所有图像处理完成！")

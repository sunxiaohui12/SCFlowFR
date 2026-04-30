import os
from PIL import Image
import sys


def check_png_files(folder_path):
    # 确保文件夹路径存在
    if not os.path.isdir(folder_path):
        print(f"错误: {folder_path} 不是一个有效的文件夹路径")
        return

    # 获取文件夹中所有文件
    png_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".png")]

    if not png_files:
        print("文件夹中没有找到 PNG 文件")
        return

    print(f"找到 {len(png_files)} 个 PNG 文件，开始检查...")

    # 用于记录损坏的文件
    corrupted_files = []

    # 检查每个 PNG 文件
    for file_name in png_files:
        file_path = os.path.join(folder_path, file_name)
        try:
            # 尝试打开和验证 PNG 文件
            with Image.open(file_path) as img:
                img.verify()  # 验证文件完整性
            print(f"文件 {file_name}: 完整")
        except (IOError, SyntaxError, Image.DecompressionBombError) as e:
            print(f"文件 {file_name}: 损坏或截断 ({str(e)})")
            corrupted_files.append(file_name)

    # 总结报告
    print("\n检查完成！")
    if corrupted_files:
        print(f"发现 {len(corrupted_files)} 个损坏的 PNG 文件:")
        for file in corrupted_files:
            print(f"- {file}")

        # 询问用户是否删除损坏的文件
        while True:
            response = input("\n是否删除所有损坏的 PNG 文件？(y/n): ").lower()
            if response in ["y", "n"]:
                break
            print("请输入 'y' 或 'n'")

        if response == "y":
            for file_name in corrupted_files:
                file_path = os.path.join(folder_path, file_name)
                try:
                    os.remove(file_path)
                    print(f"已删除: {file_name}")
                except OSError as e:
                    print(f"删除 {file_name} 失败: {str(e)}")
            print("删除操作完成")
        else:
            print("未删除任何文件")
    else:
        print("所有 PNG 文件都完整")


if __name__ == "__main__":
    # 检查是否提供了命令行参数
    if len(sys.argv) != 2:
        print("用法: python check_file.py <文件夹路径>")
    else:
        folder_path = sys.argv[1]
        check_png_files(folder_path)

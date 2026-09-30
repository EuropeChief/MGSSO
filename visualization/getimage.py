import os
import shutil

data_dir = '/data2/datasets/coco/'  # COCO 数据集根目录
save_dir = '/data4/2022/gym/OFA/visualization/image/'  # 目标保存目录
image_id_file = '/data4/2022/gym/OFA/visualization/image_id.txt'  # 存放 image_id 的文件
splits = ['train2014', 'val2014', 'test2014']  # 可搜索的文件夹

def get_image_path(image_id):
    """在所有数据集 split 中查找图片"""
    for split in splits:
        img_filename = f"COCO_{split}_{int(image_id):012d}.jpg"  # COCO 格式文件名
        img_path = os.path.join(data_dir, split, img_filename)
        if os.path.exists(img_path):
            return img_path, img_filename
    return None, None

def download_image(img_path, img_filename, image_id):
    """复制图片到目标目录"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    save_path = os.path.join(save_dir, img_filename)
    shutil.copy(img_path, save_path)
    print(f"{image_id} saved successfully")

def process_images():
    """读取 image_id 文件并处理图片"""
    if not os.path.exists(image_id_file):
        print(f"Image ID file {image_id_file} not found.")
        return
    
    with open(image_id_file, 'r') as f:
        image_ids = [line.strip() for line in f]
    
    for image_id in image_ids:
        img_path, img_filename = get_image_path(image_id)
        if img_path:
            download_image(img_path, img_filename, image_id)
        else:
            print(f"{image_id} not found")

if __name__ == "__main__":
    process_images()

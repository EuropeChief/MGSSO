from pycocotools.coco import COCO
import os

def get_caption(image_filename):
    # COCO的train和val标注文件
    annotation_files = [
        '/data2/datasets/coco/annotations/captions_train2014.json',
        '/data2/datasets/coco/annotations/captions_val2014.json'
    ]

    # 遍历所有标注文件
    for annotation_file in annotation_files:
        if not os.path.exists(annotation_file):
            print(f"标注文件 {annotation_file} 不存在.")
            continue

        # 加载COCO数据
        coco = COCO(annotation_file)

        # **查找 image_id**
        image_id = None
        for img in coco.dataset["images"]:
            if img["file_name"] == image_filename:
                image_id = img["id"]
                break  # 找到即停止

        if image_id is None:
            print(f"图片 {image_filename} 在 {annotation_file} 中未找到.")
            continue

        # **获取该图片的所有 caption**
        caption_ids = coco.getAnnIds(imgIds=[image_id])
        captions = coco.loadAnns(caption_ids)

        if captions:
            print(f"\n图片 {image_filename} 的 Caption（来自 {annotation_file}）：")
            for caption in captions:
                print(f"- {caption['caption']}")
            return  # 找到后退出函数
        else:
            print(f"图片 {image_filename} 在 {annotation_file} 中没有找到 caption.")
            return

    print(f"\n没有找到图片 {image_filename} 的标注。")


# **输入图片名**
image_filename = input("请输入图片名（如：COCO_train2014_000000123456.jpg 或 COCO_val2014_000000123456.jpg）：")
get_caption(image_filename)

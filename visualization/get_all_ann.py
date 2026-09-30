import os
import pandas as pd
from pycocotools.coco import COCO

def load_image_ids(file_path):
    """ 从文本文件中读取图片 ID """
    with open(file_path, 'r') as f:
        image_ids = [int(line.strip()) for line in f if line.strip().isdigit()]
    return image_ids

def get_captions_for_image_ids(image_ids, annotation_files):
    """ 获取指定图片 ID 的 captions """
    results = []
    for annotation_file in annotation_files:
        if not os.path.exists(annotation_file):
            print(f"标注文件 {annotation_file} 不存在.")
            continue
        
        coco = COCO(annotation_file)
        for image_id in image_ids:
            caption_ids = coco.getAnnIds(imgIds=[image_id])
            captions = coco.loadAnns(caption_ids)
            
            if captions:
                caption_texts = [caption['caption'] for caption in captions]
                results.append((image_id, "\n".join(caption_texts[:5])))  # 只取前5个 caption，并换行显示
            else:
                print(f"图片 ID {image_id} 在 {annotation_file} 中没有找到 caption.")
    
    return results

def save_to_excel(results, output_file):
    """ 将结果保存为 Excel 文件 """
    df = pd.DataFrame(results, columns=['image_id', 'captions'])
    df.to_excel(output_file, index=False)

def main():
    image_id_file = '/data4/2022/gym/OFA/visualization/image_id.txt'
    annotation_files = [
        '/data2/datasets/coco/annotations/captions_train2014.json',
        '/data2/datasets/coco/annotations/captions_val2014.json'
    ]
    output_file = 'coco_captions.xlsx'
    
    image_ids = load_image_ids(image_id_file)
    results = get_captions_for_image_ids(image_ids, annotation_files)
    save_to_excel(results, output_file)
    print(f"结果已保存至 {output_file}")

if __name__ == '__main__':
    main()

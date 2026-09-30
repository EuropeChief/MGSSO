import json
import pandas as pd
import os

def json_to_excel(json_file, excel_file):
    # 读取 JSON 文件
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 转换为 DataFrame
    df = pd.DataFrame(data)
    
    # 获取当前目录并保存 Excel 文件
    current_dir = os.getcwd()
    excel_path = os.path.join(current_dir, excel_file)
    df.to_excel(excel_path, index=False)
    
    # 统计 image_id 总数
    total_images = df['image_id'].nunique()
    print(f"Total unique image_id: {total_images}")
    print(f"Excel file saved at: {excel_path}")

# 示例调用
json_to_excel('/data4/2022/gym/OFA/results/caption/stage2_improved_20250106/test_predict.json', 'test_predict.xlsx')

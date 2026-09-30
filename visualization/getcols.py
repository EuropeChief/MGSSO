import pandas as pd

# 指定 TSV 文件路径
file_path = "/data4/2022/gym/OFA/dataset/Flickr30k/caption_test_flickr1.tsv"

# 读取 TSV 文件
df = pd.read_csv(file_path, sep="\t")

# 获取所有列名
columns = df.columns.tolist()

# 获取第一行数据
first_row = df.iloc[0].tolist()

# 打印结果
print("列名:", columns)
print("第一行数据:", first_row)

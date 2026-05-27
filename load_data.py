# from datasets import load_dataset

# # Login using e.g. `huggingface-cli login` to access this dataset
# # ds = load_dataset("redlessone/Derm1M")
# ds = load_dataset(
#     "redlessone/Derm1M",
#     trust_remote_code=True,  # 必须保留，自定义数据集需要
#     cache_dir='/datasets_hdd2/yjzhang/data/Derm1M'    # 指定自定义存储路径
# )


from datasets import Dataset

# 替换成你的arrow文件绝对路径
ARROW_FILE_PATH = "/datasets_hdd2/yjzhang/data/Derm1M/redlessone___derm1_m/default/0.0.0/3c1da1beeae41027b73d5aa8e9b4ec4c5dc83ad7/derm1_m-valid.arrow"

# 关键：用datasets的Dataset.from_file读取（专门解析自定义arrow文件）
try:
    dataset = Dataset.from_file(ARROW_FILE_PATH)
    df = dataset.to_pandas()
    
    print("✅ 单个arrow文件读取成功！")
    print(f"数据形状：{df.shape}")
except Exception as e:
    print(f"❌ 读取失败：{e}")
    print("→ 检查：文件路径是否正确，是否是datasets库生成的arrow文件")
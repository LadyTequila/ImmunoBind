import chardet
import os
import pandas as pd

def detect_encoding(file_path):
    try:
        with open(file_path, 'rb') as f:
            raw_data = f.read()
            result = chardet.detect(raw_data)
            return result['encoding']
    except FileNotFoundError:
        print(f"文件 {file_path} 未找到。")
        return None
    except Exception as e:
        print(f"编码检测时出现错误: {e}")
        return None

def convert_to_utf8_and_delete_columns(input_file, output_file, columns_to_delete):
    original_encoding = detect_encoding(input_file)
    if original_encoding is None:
        return
    try:
        file_size = os.path.getsize(input_file)
        processed_size = 0

        # 使用 pandas 读取文件
        df = pd.read_csv(input_file, sep='\t', encoding=original_encoding)

        # 删除指定列
        if columns_to_delete:
            df = df.drop(columns=columns_to_delete, errors='ignore')

        # 将处理后的数据保存为 UTF-8 编码的 TSV 文件，设置 index=False 避免写入索引
        df.to_csv(output_file, sep='\t', encoding='utf-8', na_rep='nan', index=False)

        print(f"\n文件 {input_file} 已成功转换为 UTF-8 编码，删除指定列后保存到 {output_file}。")
    except UnicodeDecodeError:
        print(f"使用检测到的编码 {original_encoding} 无法正确读取文件，请手动指定编码。")
    except Exception as e:
        print(f"文件转换过程中出现错误: {e}")

input_file = "C:/Users/21636/Desktop/ImmunoBind/data/SearchTable-2024-12-15 09_08_53.829.tsv"  # 输入文件路径
output_file = "C:/Users/21636/Desktop/ImmunoBind/data/tcr.tsv"  # 输出文件路径
columns_to_delete = ['V', 'J','MHC A','MHC B']

convert_to_utf8_and_delete_columns(input_file, output_file, columns_to_delete)
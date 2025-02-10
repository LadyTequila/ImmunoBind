import chardet

# 检测原始文件的编码
def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
        return result['encoding']

# 转换文件编码为 UTF-8
def convert_to_utf8(input_file, output_file):
    original_encoding = detect_encoding(input_file)
    with open(input_file, 'r', encoding=original_encoding) as infile:
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for line in infile:
                outfile.write(line)

input_file = "C:/Users/21636/Desktop/ImmunoBind/data/SearchTable-2024-12-15 09_08_53.829.tsv"  # 输入文件路径
output_file = "C:/Users/21636/Desktop/ImmunoBind/data/tcr.tsv"  # 输出文件路径

convert_to_utf8(input_file, output_file)

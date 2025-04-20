import chardet

file_path = r"c:\Users\jwolf\Documents\Coding-Projects\HebHTR\model\wordCharList.txt"

with open(file_path, 'rb') as f:
    raw_data = f.read()
    result = chardet.detect(raw_data)
    print(f"Detected encoding: {result['encoding']}")
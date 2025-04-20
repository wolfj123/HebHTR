input_file = r"c:\Users\jwolf\Documents\Coding-Projects\HebHTR\model\wordCharList.txt"
output_file = r"c:\Users\jwolf\Documents\Coding-Projects\HebHTR\model\wordCharList_utf8.txt"

# Replace 'detected_encoding' with the encoding detected in Step 1
detected_encoding = 'iso-8859-8'

with open(input_file, 'r', encoding=detected_encoding) as infile:
    content = infile.read()

with open(output_file, 'w', encoding='utf-8') as outfile:
    outfile.write(content)

print(f"File converted to UTF-8 and saved as {output_file}")
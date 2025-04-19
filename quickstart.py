from HebHTR import *

# Create new HebHTR object.
img = HebHTR('input/image.png')

# Infer words from image.
# text = img.imgToWord(iterations=5, decoder_type='word_beam')
text = img.imgToWord(iterations=5, decoder_type='best_path')


print(text)
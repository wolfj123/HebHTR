from HebHTR import *

# Create new HebHTR object.
img = HebHTR('input/image.png')

# Infer words from image.
# text = img.imgToWord(iterations=5, decoder_type='word_beam')
text = img.imgToWord(iterations=5, decoder_type='word_beam')


print(text)
from HebHTR import *

# Create new HebHTR object.
img = HebHTR('example.png')

# Infer words from image.
text = img.imgToWord(iterations=5, decoder_type='word_beam')
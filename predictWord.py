from Model import Model, DecoderType
from processFunctions import preprocessImageForPrediction
import numpy as np

class Batch:
    def __init__(self, gtTexts, imgs):
        self.imgs = np.stack(imgs, axis=0)
        self.gtTexts = gtTexts

class FilePaths:
    fnCharList = 'model/charList.txt'
    fnCorpus = 'data/corpus.txt'

def infer(model, image):
    # Use the imgSize from the model instance
    img = preprocessImageForPrediction(image, model.imgSize)
    y_pred = model.model.predict(img)
    input_lengths = [y_pred.shape[1]]  # Length of the sequence
    decoded = model.decode(y_pred, input_lengths)
    return decoded

def getModel(decoder_type):
    with open('./model/charList.txt', 'r', encoding='utf-8') as f:
        charList = list(f.read().strip())

    # Define the required parameters
    imgSize = (128, 32)  # Example image size (height, width)
    maxTextLen = 32      # Example maximum text length

    # Map decoder_type string to DecoderType enum
    decoderType = DecoderType.BestPath if decoder_type == 'best_path' else DecoderType.WordBeamSearch

    # Create and return the Model instance
    model = Model(charList, batchSize=1, imgSize=imgSize, maxTextLen=maxTextLen, decoderType=decoderType, mustRestore=False)
    return model


def predictWord(image, model):
    return infer(model, image)


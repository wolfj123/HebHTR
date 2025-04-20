import cv2
import numpy as np

def preprocessImageForPrediction(img, imgSize):
    """Preprocess the image for prediction."""
    # Convert to grayscale if the image has 3 channels (RGB)
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    (h, w) = img.shape

    # Resize the image to the target size while maintaining aspect ratio
    fx = w / imgSize[0]
    fy = h / imgSize[1]
    f = max(fx, fy)
    newSize = (max(min(imgSize[0], int(w / f)), 1), max(min(imgSize[1], int(h / f)), 1))
    img = cv2.resize(img, newSize, interpolation=cv2.INTER_CUBIC)

    # Create a blank image and place the resized image in the center
    target = np.ones([imgSize[1], imgSize[0]]) * 255
    target[0:newSize[1], 0:newSize[0]] = img

    # Normalize the image
    img = target / 255.0
    img = img.T
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)
    return img
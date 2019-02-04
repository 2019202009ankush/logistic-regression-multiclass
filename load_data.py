import numpy as np
import matplotlib.pyplot as plt

def loadData(imagesFilename, labelsFilename):
    """
    Reads the MINST's dataset in a 
    numpy array of float32 type.
    As is stated in the minst website
    the integers in the files are stored 
    in the MSB first (high endian) so
    there is needed some manipulation
    to extract information from the 
    image header.
    

    Arguments:
    filename -- String with the data filename

    Return:
    a tuple with the images loaded in a numpy array
    of shape (60000, 28, 28) for train images dataset, and 
    the labels numpy array of shape (60000, ) for train labels, 
    or (10000, 28, 28) and (10000, ) on the test dataset.
    """
    offsetImages = 4
    offsetLabels = 2
    intType = np.dtype( 'int32' ).newbyteorder( '>' )
    nMetaDataBytesImages = offsetImages * intType.itemsize
    nMetaDataBytesLabels = offsetLabels * intType.itemsize
    imageData = np.fromfile(imagesFilename, dtype = 'ubyte' )
    magicBytes, nImages, width, height = np.frombuffer(imageData[:nMetaDataBytesImages].tobytes(), intType )
    imageData = imageData[nMetaDataBytesImages:].astype( dtype = 'float64' ).reshape([nImages, width, height])
    imageLabels = np.fromfile(labelsFilename, dtype = 'ubyte' )[offsetLabels * intType.itemsize:]

    return imageData, imageLabels


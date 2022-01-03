import cv2
import numpy as np

class CropLayer(object):
    def __init__(self, params, blobs):
        self.xstart = 0
        self.xend = 0
        self.ystart = 0
        self.yend = 0

    # Our layer receives two inputs. We need to crop the first input blob
    # to match a shape of the second one (keeping batch size and number of channels)
    def getMemoryShapes(self, inputs):
        inputShape, targetShape = inputs[0], inputs[1]
        batchSize, numChannels = inputShape[0], inputShape[1]
        height, width = targetShape[2], targetShape[3]

        self.ystart = int((inputShape[2] - targetShape[2]) / 2)
        self.xstart = int((inputShape[3] - targetShape[3]) / 2)
        self.yend = self.ystart + height
        self.xend = self.xstart + width

        return [[batchSize, numChannels, height, width]]

    def forward(self, inputs):
        return [inputs[0][:,:,self.ystart:self.yend,self.xstart:self.xend]]

cv2.dnn_registerLayer('Crop', CropLayer)

class HedLayer():
    def __init__(self) -> None:
        self.net = cv2.dnn.readNetFromCaffe("dl_model/deploy.prototxt", "dl_model/hed_pretrained_bsds.caffemodel")

    def forward(self, image):
        image = cv2.resize(image, (image.shape[1], image.shape[0]))
        inp = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(image.shape[1], image.shape[0]),
                                mean=(104.00698793, 116.66876762, 122.67891434),
                                swapRB=False, crop=False)
        self.net.setInput(inp)
        out = self.net.forward()
        out = out[0, 0]
        out = cv2.resize(out, (image.shape[1], image.shape[0]))
        out = 255 * out
        out = out.astype(np.uint8)
        return out
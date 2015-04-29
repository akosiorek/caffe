import caffe
import numpy as np


MODEL_FILE = '../models/segmentation/deploy.prototxt'
PRETRAINED = '../models/segmentation/snapshot/train_iter_2000.caffemodel'
IMAGE_FILE = '/home/adam/workspace/segmentation/data/weizmann_horse_db/rgb_64x48/horse065.jpg'
MEAN_FILE = np.load('../data/segmentation/horses_mean.npy')
INPUT_SCALE = 1.0/255

class Segmenter(caffe.Net):
    """
    Segmenter extends Net for image segmentation.
    """
    def __init__(self, model_file, pretrained_file, mean=None, raw_scale=None):
        caffe.Net.__init__(self, model_file, pretrained_file, caffe.TEST)

        # configure pre-processing
        in_ = self.inputs[0]
        self.transformer = caffe.io.Transformer(
            {in_: self.blobs[in_].data.shape for in_ in self.inputs})
        self.transformer.set_transpose(in_, (2,0,1))
        if mean is not None:
            self.transformer.set_mean(in_, mean)
        if raw_scale is not None:
            self.transformer.set_raw_scale(in_, raw_scale)


        self.image_dims = np.array(self.blobs[in_].data.shape[2:])
        print self.blobs[in_].data.shape, self.image_dims

    def predict(self, inputs):
        """
        Predict segmentation inputs.

        Take
        inputs: iterable of (H x W x K) input ndarrays.

        Give
        predictions: (N x H x W) ndarray of image segmentations
                     for N images and H x W images.
        """
        # Scale to standardize input dimensions.
        input_ = np.zeros((len(inputs),
            self.image_dims[0], self.image_dims[1], inputs[0].shape[2]),
            dtype=np.float32)
        for ix, in_ in enumerate(inputs):
            input_[ix] = caffe.io.resize_image(in_, self.image_dims)

        # Classify
        caffe_in = np.zeros(np.array(input_.shape)[[0,3,1,2]],
                            dtype=np.float32)
        for ix, in_ in enumerate(input_):
            caffe_in[ix] = self.transformer.preprocess(self.inputs[0], in_)
        out = self.forward_all(**{self.inputs[0]: caffe_in})
        predictions = out[self.outputs[0]]

        return predictions


segmenter = Segmenter(MODEL_FILE, PRETRAINED, MEAN_FILE, INPUT_SCALE);

img = caffe.io.load_image(IMAGE_FILE)

img = segmenter.predict([img,])
img = np.reshape(img, img.shape[2:])

print img.shape
import matplotlib.pyplot as plt


print 'mean:', np.mean(img)
print 'max:', np.max(img)
print 'min:', np.min(img)
plt.figure()
plt.imshow(img)
plt.show()

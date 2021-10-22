from PIL.Image import Image
import numpy as np
import tensorflow as tf

class ImageLoader():
    IMG_SIZE=160

    def __init__(self,img_size):
        self.IMG_SIZE = img_size

    def get_formatted_image_from_path(self, path):
        pic = Image.open(path)
        pic = pic.convert("RGB") #L for grayscale
        pix = self.to_numpy(pic)
        pix = self.format_example(pix)

        return pix

    def to_numpy(self, im):
        im.load()
        # unpack data
        e = Image._getencoder(im.mode, 'raw', im.mode)
        e.setimage(im.im)

        # NumPy buffer for the result
        shape, typestr = Image._conv_type_shape(im)
        data = np.empty(shape, dtype=np.dtype(typestr))
        mem = data.data.cast('B', (data.data.nbytes,))

        bufsize, s, offset = 65536, 0, 0
        while not s:
            l, s, d = e.encode(bufsize)
            mem[offset:offset + len(d)] = d
            offset += len(d)
        if s < 0:
            raise RuntimeError("encoder error %d in tobytes" % s)
        return data

    def format_example(self, image):
        image = tf.cast(image, tf.float32)
        image = (image / 127.5) - 1
        image = tf.image.resize(image, (self.IMG_SIZE, self.IMG_SIZE))
        return image
import numpy as np
from PIL import Image
from . import util

def load_image(path, channel_format = util.default_format, dtype=np.float32):
    data = np.array(Image.open(path), dtype=dtype)
    return util.adapt_channelorder(data, channel_format)
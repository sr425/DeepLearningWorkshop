import numpy as np
from glob import glob

default_format = 'channels_last'

def adapt_channelorder(data, channel_format):
    if channel_format == 'channels_last' or channel_format == 'NHWC':
        return data
    else:
        return np.transpose(data, [2, 0, 1])
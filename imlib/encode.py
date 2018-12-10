from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io

import numpy as np

from imlib.dtype import im2uint, uint2im
from PIL import Image


def imencode(image, format='PNG', quality=95):
    """Encode an [-1.0, 1.0] image into byte string.

    Args:
        format  : 'PNG' or 'JPEG'.
        quality : Only for 'JPEG'.

    Returns:
        Byte string.
    """
    byte_io = io.BytesIO()
    image = Image.fromarray(im2uint(image))
    image.save(byte_io, format=format, quality=quality)
    bytes = byte_io.getvalue()
    return bytes


def imdecode(bytes):
    """Decode byte string to float64 image in [-1.0, 1.0].

    Args:
        bytes: Byte string.

    Returns:
        A float64 image in [-1.0, 1.0].
    """
    byte_io = io.BytesIO()
    byte_io.write(bytes)
    image = np.array(Image.open(byte_io))
    image = uint2im(image)
    return image

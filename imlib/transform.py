from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import skimage.color as color
import skimage.transform as transform


rgb2gray = color.rgb2gray
gray2rgb = color.gray2rgb

imresize = transform.resize
imrescale = transform.rescale


def immerge(images, n_row=None, n_col=None, padding=0, pad_value=0):
    """Merge images to an image with (n_row * h) * (n_col * w).

    `images` is in shape of N * H * W(* C=1 or 3)
    """
    n = images.shape[0]
    if n_row:
        n_row = max(min(n_row, n), 1)
        n_col = int(n - 0.5) // n_row + 1
    elif n_col:
        n_col = max(min(n_col, n), 1)
        n_row = int(n - 0.5) // n_col + 1
    else:
        n_row = int(n ** 0.5)
        n_col = int(n - 0.5) // n_row + 1

    h, w = images.shape[1], images.shape[2]
    shape = (h * n_row + padding * (n_row - 1),
             w * n_col + padding * (n_col - 1))
    if images.ndim == 4:
        shape += (images.shape[3],)
    img = np.full(shape, pad_value, dtype=images.dtype)

    for idx, image in enumerate(images):
        i = idx % n_col
        j = idx // n_col
        img[j * (h + padding):j * (h + padding) + h,
            i * (w + padding):i * (w + padding) + w, ...] = image

    return img

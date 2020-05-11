import numbers
import skimage
import numpy
import skimage.filters.rank.generic

from skimage import data

import dask_ndfilters._utils as _utils

@_utils._update_wrapper(skimage.filters.rank.generic.entropy)
def entropy(image, selem, out=None, mask=None, shift_x=False, shift_y=False):
    selem = np.ones(2 * [entropy_filter_size])
    
    depth = 5
    depth, boundary = _utils._get_depth_boundary(image.ndim, depth, "none")
    
    if (image.ndim == 2):
         result = image.map_overlap(
            skimage.filters.rank.generic.entropy,
            depth = depth,
            boundary = boundary,
            dtype = image.dtype,
            selem = selem,
            out = out,
            mask = mask,
            shift_x = shift_x,
            shift_y = shift_y
        )
    
    else:
        lst = []
        for i in range(np.size(stack, axis = 0)):
            lst.append(entropy(stack[i], selem))
        result = np.stack(lst, axis =0)
            
    return result

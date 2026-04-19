import numpy as np
import cupy as cp
import cv2
import time
from cupy.cuda import texture
from cupy.cuda import runtime
import glfw
from zero_copy_window import ZeroCopyWindow
def create_texture_object(img_cp):
    h, w, c = img_cp.shape
    bytes_per_pixel = 16 
    alignment = 256
    pitch_bytes = ((w * bytes_per_pixel + alignment - 1) // alignment) * alignment
    padded_w = pitch_bytes // bytes_per_pixel
    rgba = cp.zeros((h, padded_w, 4), dtype=cp.float32)
    rgba[:, :w, :3] = img_cp
    ch_fmt = texture.ChannelFormatDescriptor(32, 32, 32, 32, runtime.cudaChannelFormatKindFloat)
    res_ptr = texture.ResourceDescriptor(
        runtime.cudaResourceTypePitch2D, 
        arr=rgba,                  
        chDesc=ch_fmt, 
        width=w,
        height=h,
        pitchInBytes=pitch_bytes
    )
    tex_ptr = texture.TextureDescriptor(
        addressModes=(runtime.cudaAddressModeClamp, runtime.cudaAddressModeBorder),borderColors = (0.0,0.0,0.0,0.0),
        filterMode=runtime.cudaFilterModeLinear,
        readMode=runtime.cudaReadModeElementType
    )
    tex_obj = texture.TextureObject(res_ptr, tex_ptr)
    return tex_obj, rgba

# tex_handle, _internal_storage = create_texture_object(img) # img要在显卡里面



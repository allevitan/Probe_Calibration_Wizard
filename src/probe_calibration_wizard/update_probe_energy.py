from cdtools.tools.propagators import *
import torch as t


def pad_by_factor(im, factor, n_dims=2):
    im_shape = t.as_tensor(im.shape[-n_dims:], dtype=t.float32)
    pad_to = t.round(im_shape * factor).to(dtype=t.int32)
    if t.any(pad_to < im_shape.to(dtype=t.int32)):
        raise ValueError(
            'Cannot pad by a factor less than 1, use crop_by_factor instead')
    # Note - this is carefully set up so that the zero frequency pixels that
    # match the standard definition of the fftshift will stay at the zero
    # frequency location. Don't mess with this if you haven't thought it
    # through!
    left_paddings = (t.div(pad_to, 2, rounding_mode='floor') -
                     t.div(im_shape, 2, rounding_mode='floor'))
    right_paddings = pad_to - im_shape - left_paddings
    pad_list = []
    for lp, rp in zip(left_paddings, right_paddings):
        pad_list.extend([int(rp), int(lp)])
    # The convention for pytorch is to start from the left pad on the final
    # dimension and then go backward through the dimensions.
    pad_list = pad_list[::-1]
    return t.nn.functional.pad(im, pad_list)


def crop_by_factor(im, factor, n_dims=2):
    im_shape = t.as_tensor(im.shape[-n_dims:], dtype=t.float32)
    crop_to = t.round(im_shape * factor).to(dtype=t.int32)
    if t.any(crop_to > im_shape.to(dtype=t.int32)):
        raise ValueError(
            'Cannot crop by a factor greater than 1, use pad_by_factor instead')
    # Note - this is carefully set up so that the zero frequency pixels that
    # match the standard definition of the fftshift will stay at the zero
    # frequency location. Don't mess with this if you haven't thought it
    # through!
    left_crops = (t.div(im_shape, 2, rounding_mode='floor') -
                  t.div(crop_to, 2, rounding_mode='floor'))
    right_crops = im_shape - crop_to - left_crops
    sl = (Ellipsis, ) + \
        tuple((slice(int(lc), int(rc)) if rc != 0 else slice(int(lc),None))
              for lc, rc in zip(left_crops, -right_crops))

    return im[sl]    


def pad_or_crop_by_factor(im, factor, n_dims=2):
    if factor > 1:
        return pad_by_factor(im, factor, n_dims=n_dims)
    else:
        return crop_by_factor(im, factor, n_dims=n_dims)

    
def change_energy(probe, energy_factor):
    # I want a probe that is rescaled in the same way that the diffraction
    # pattern of a probe from a fixed diffractive optic would be if the
    # energy changes by a factor of energy_factor. The resulting probe
    # is reported in the detector conjugate coordinates for the same
    # detector at the new energy.
    fourier_probe = far_field(probe)
    fourier_probe = pad_or_crop_by_factor(fourier_probe, energy_factor)
    real_probe = inverse_far_field(fourier_probe)

    # It's an edge case, but sometimes 1/energy_factor isn't rounded the same
    # way as energy_factor, so it's best to do the second crop using the
    # actual realized difference in the shapes

    # NOTE: This is likely to fail on non-square probes
    cropped = pad_or_crop_by_factor(real_probe, probe.shape[-1]/real_probe.shape[-1])
    return cropped

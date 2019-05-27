# Main references:
# https://github.com/Peter554/StainTools

import numpy as np
import cv2 as cv


__all__ = ["BrightnessStandardizer"]


# ###################### Exceptions #############################


class TissueException(Exception):
    def __init__(self, *args, **kwargs):
        super(TissueException, self).__init__(*args, **kwargs)


# ###################### MISC ##################################

def remove_zeros(I):
    """
    Remove zeros in an array, replace with 1's.

    :param I: Array.
    :return: New array where 0's have been replaced with 1's.
    """
    mask = (I == 0.)
    I[mask] = 1.
    return I


def convert_RGB_to_OD(I):
    """
    Convert from RGB to optical density (OD_RGB) space.

    RGB = 255 * exp(-1 * OD_RGB)

    :param I: Image RBG uint8.
    :return: Optical density RGB image.
    """
    I = remove_zeros(I)
    eps = 1e-6  # for numerical stability.
    return np.maximum(-1 * np.log(I / 255), eps)


def convert_OD_to_RGB(OD):
    """
    Convert from optical density (OD_RGB) into RGB.

    RGB = 255 * exp(-1 * OD_RGB)

    :param OD: Optical density RGB image.
    :return: Image RGB uint8.
    """
    assert OD.min() >= 0, "Negative optical density."

    eps = 1e-6  # For numerical stability.
    OD = np.maximum(OD, eps)
    return (255 * np.exp(-1 * OD)).astype(np.uint8)

def normalize_rows(A):
    """
    Normalize the rows of an array.

    :param A: Array.
    :return: Array with rows have been normalized.
    """
    return A / np.linalg.norm(A, axis=1)[:, None]


def get_sign(x):
    """
    Returns the sign of x.

    :param x: A scalar.
    :return: The sign of x \in (+1, -1, 0).
    """
    if x > 0:
        return +1
    elif x < 0:
        return -1
    elif x ==0:
        return 0


def array_equal(A, B, eps=1e-6):
    """
    Are arrays A and B equal?

    :param A: Array.
    :param B: Array
    :param eps: Tolerance.
    :return: True/False.
    """
    if A.ndim != B.ndim:
        return False
    if A.shape != B.shape:
        return False
    if np.min(np.abs(A - B)) > eps:
        return False
    return True

# ########################## IMAGE CHECKS #####################

def is_image(x):
    """
    Is x an image?
    i.e. numpy array of 2 or 3 dimensions.

    :param x: Input.
    :return: True/False.
    """
    if not isinstance(x, np.ndarray):
        return False
    if x.ndim not in [2, 3]:
        return False
    return True


def is_gray_image(x):
    """
    Is x a gray image?

    :param x: Input.
    :return: True/False.
    """
    if not is_image(x):
        return False
    squeezed = x.squeeze()
    if not squeezed.ndim == 2:
        return False
    return True


def is_uint8_image(x):
    """
    Is x a uint8 image?

    :param x: Input.
    :return: True/False.
    """
    if not is_image(x):
        return False
    if x.dtype != np.uint8:
        return False
    return True


def check_image_and_squeeze_if_gray(I):
    """
    Check that I is an image and squeeze to 2D if it is gray.

    :param I: Input.
    :return: Squeezed input.
    """
    assert is_image(I), "I should be an image (2D or 3D numpy array)."

    if is_gray_image(I):
        return I.squeeze()
    else:
        return I


# ########################## TISSUE MASK #######################

def get_tissue_mask(I, luminosity_threshold=0.8):
    """
    Get a binary mask where true denotes pixels with a luminosity less than the specified threshold.
    Typically, we use to identify tissue in the image and exclude the bright white background which is many
    due to the light.

    :param I: RGB uint8 image.
    :param luminosity_threshold: Luminosity threshold.
    :return: Binary mask.
    """
    assert is_uint8_image(I), "Image should be RGB uint8."
    I_LAB = cv.cvtColor(I, cv.COLOR_RGB2LAB)
    L = I_LAB[:, :, 0] / 250.  # convert to range [0, 1]
    mask = L < luminosity_threshold

    # check if it's not empty.
    if mask.sum() == 0:
        raise TissueException("Empty tissue mask was computed.")

    return mask


# ########################## BRIGHTNESS STANDARIZATION #######################

class BrightnessStandardizer(object):
    """
    A class for standardizing image brightness. This can improve performance of other normalizers.
    """
    def __init__(self, percentile=95):
        self.percentile = percentile

    def transform(self, I, percentile=None):
        """
        Transform image I to standard brightness.
        Modifies the luminosity channel such that a fixed percentile is saturated.

        :param I: Image uint8 RGB.
        :param percentile: Percentile for luminosity saturation. At least (100 - percentile)% of pixels should be
               fully luminous (white).
        :return: Image uint8 RGB with standardized brightness.
        """
        assert is_uint8_image(I), "Image should be RGB uint8."

        if not percentile:
            percentile = self.percentile

        I_LAB = cv.cvtColor(I, cv.COLOR_RGB2LAB)
        L_float = I_LAB[:, :, 0].astype(float)
        p = np.percentile(L_float, percentile)
        I_LAB[:, :, 0] = np.clip(255 * L_float / p, 0, 255).astype(np.uint8)
        I = cv.cvtColor(I_LAB, cv.COLOR_LAB2RGB)
        return I

    def __call__(self, I):
        return self.transform(I, self.percentile)

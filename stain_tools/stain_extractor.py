# Main references:
# https://github.com/Peter554/StainTools

from abc import ABC, abstractmethod

import numpy as np

import spams
import cv2 as cv

from stain_tools.tools import convert_RGB_to_OD, get_tissue_mask, normalize_rows, is_uint8_image


class StainExtractor(ABC):
    """
    A stain extractor provides a method for estimating the stain matrix and concentration matrix.
    """
    @abstractmethod
    def get_stain_matrix(self, I, *args):
        """
        Estimate stain matrix given and image and relevant method parameters.
        """

    @staticmethod
    def get_concentrations(I, stain_matrix, **kwargs):
        """
        Estimate concentration matrix given an image, stain matrix and relevant method parameters.
        """
        OD = convert_RGB_to_OD(I).reshape((-1, 3))
        lasso_regularizer = kwargs["lasso_regularizer"] if "lasso_regularizer" in kwargs.keys() else 0.01
        return spams.lasso(X=OD.T, D=stain_matrix.T, mode=2, lambda1=lasso_regularizer, pos=True).toarray().T


class MacenkoStainExtractor(StainExtractor):
    """
    Stain matrix estimation via method of:
    M. Macenko et al.,
    'A method for normalizing histology slides for quantitative analysis',
    https://ieeexplore.ieee.org/document/5193250
    """
    @staticmethod
    def get_stain_matrix(I, luminosity_threshold=0.8, angular_percentile=99):
        """
        Estimate the stain matrix using the aforementioned method.
        :param I: Image RGB uint8.
        :param luminosity_threshold:
        :param angular_percentile:
        :return: H&E stain matrix.
        """
        tissue_mask = get_tissue_mask(I, luminosity_threshold=luminosity_threshold).reshape((-1,))

        OD = convert_RGB_to_OD(I).reshape((-1, 3))
        OD = OD[tissue_mask]

        # Eigenvectors of cov in OD space (orthogonal as cov symmetric)
        _, V = np.linalg.eigh(np.cov(OD, rowvar=False))
        # the two principle eigenvectors
        V = V[:, [2, 1]]
        # make sure that vectors are pointing the right way
        if V[0, 0] < 0:
            V[:, 0] *= -1
        if V[0, 1] < 0:
            V[:, 1] *= -1
        # project into this basis.
        pro = np.dot(OD, V)

        # Angular coordinates with respect to the principle, orthogonal eigenvectors
        phi = np.arctan2(pro[:, 1], pro[:, 0])
        # min and max angles
        minPhi = np.percentile(phi, 100 - angular_percentile)
        maxPhi = np.percentile(phi, angular_percentile)

        # the two principle colors
        v1 = np.dot(V, np.array([np.cos(minPhi), np.sin(minPhi)]))
        v2 = np.dot(V, np.array([np.cos(maxPhi), np.sin(maxPhi)]))

        # Order of H and E.
        # H first row.
        if v1[0] > v2[0]:
            HE = np.array([v1, v2])
        else:
            HE = np.array([v2, v1])

        return normalize_rows(HE)


class VahadaneStainExtractor(StainExtractor):
    """
    Stain matrix estimation via method of:
    A. Vahadane et al.,
    'Structure-Preserving Color Normalization and Sparse Stain Separation for Histological Images'
    https://ieeexplore.ieee.org/document/7460968
    """
    @staticmethod
    def get_stain_matrix(I, luminosity_threshold=0.8, dictionary_regularizer=0.1):
        """
        Estimate the stain matrix using the aforementioned method.
        :param I: Image RGB uint8.
        :param luminosity_threshold:
        :param dictionary_regularizer:
        :return: H&E stain matrix.
        """
        # convert to OD and ignore background.
        tissue_mask = get_tissue_mask(I, luminosity_threshold=luminosity_threshold).reshape((-1,))
        OD = convert_RGB_to_OD(I).reshape((-1, 3))
        OD = OD[tissue_mask]

        # do the dictionary learning
        dictionary = spams.trainDL(
            X=OD.T, K=2, lambda1=dictionary_regularizer, mode=2, modeD=0, posAlpha=True, posD=True, verbose=False).T

        # Order H&E.
        # H on the first row.
        if dictionary[0, 0] < dictionary[1, 0]:
            dictionary = dictionary[[1, 0], :]

        return normalize_rows(dictionary)


class ReinhardColorNormalizer(object):
    """
    Normalize a patch color to the target image using the method of:
    E. Reinhard, M. Adhikhmin, B. Gooch, and P. Shirley,
    'Color transfer between images',
    https://www.cs.tau.ac.il/~turkel/imagepapers/ColorTransfer.pdf
    """
    def __init__(self):
        self.target_means = None
        self.target_stds = None

    def fit(self, target):
        """
        Fit to a target image.

        :param target: Image RGB target.
        :return:
        """
        means, stds = self.get_mean_std(target)
        self.target_means = means
        self.target_stds = stds

    def transform(self, I):
        """
        Transform an image using the fitted statistics.
        :param I: Image RGB uint8.
        :return:
        """
        I1, I2, I3 = self.lab_split(I)
        means, stds = self.get_mean_std(I)
        norm1 = ((I1 - means[0]) * (self.target_stds[0] / stds[0])) + self.target_means[0]
        norm2 = ((I2 - means[1]) * (self.target_stds[1] / stds[1])) + self.target_means[1]
        norm3 = ((I3 - means[2]) * (self.target_stds[2] / stds[2])) + self.target_means[2]
        return self.merge_back(norm1, norm2, norm3)

    @staticmethod
    def lab_split(I):
        """
        Convert from RGB uint8 into LAB and split into channels.

        :param I: Image RGB uint8.
        :return:
        """
        assert is_uint8_image(I), "Should be an RGB uint8 image."
        I = cv.cvtColor(I, cv.COLOR_RGB2LAB)
        I_float = I.astype(np.float32)
        I1, I2, I3 = cv.split(I_float)
        I1 /= 2.55  # should now be in range [0,100]
        I2 -= 128.0  # should now be in range [-127,127]
        I3 -= 128.0  # should now be in range [-127,127]
        return I1, I2, I3

    @staticmethod
    def merge_back(I1, I2, I3):
        """
        Take separate LAB channels and merge back to give RGB uint8.
        :param I1: L.
        :param I2: A.
        :param I3: B.
        :return: Image RGB uint8.
        """
        I1 *= 2.55  # should now be in range [0,255]
        I2 += 128.0  # should now be in range [0,255]
        I3 += 128.0  # should now be in range [0,255]
        I = np.clip(cv.merge((I1, I2, I3)), 0, 255).astype(np.uint8)
        return cv.cvtColor(I, cv.COLOR_LAB2RGB)

    def get_mean_std(self, I):
        """
        Get mean and standard deviation of each channel.

        :param I: Image RGB uint8.
        :return:
        """
        assert is_uint8_image(I), "Should be an RGB uint8 image."
        I1, I2, I3 = self.lab_split(I)
        m1, sd1 = cv.meanStdDev(I1)
        m2, sd2 = cv.meanStdDev(I2)
        m3, sd3 = cv.meanStdDev(I3)
        means = m1, m2, m3
        stds = sd1, sd2, sd3
        return means, stds

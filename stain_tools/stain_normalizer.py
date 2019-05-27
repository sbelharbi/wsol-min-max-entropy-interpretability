# Main references:
# https://github.com/Peter554/StainTools

import numpy as np

from stain_tools.stain_extractor import MacenkoStainExtractor, VahadaneStainExtractor
from stain_tools.tools import convert_OD_to_RGB, get_tissue_mask


__all__ = ["StainNormalizer"]


class StainNormalizer(object):
    """
    Stain normalizer class. It holds the specified stain matrix extractor.
    """
    def __init__(self, method):
        """
        Init function.

        :param method: str, the name of the stain normalization method: "macenko", "vahadane".
        """
        if method.lower() == "macenko":
            self.extractor = MacenkoStainExtractor
        elif method == "vahadane":
            self.extractor = VahadaneStainExtractor
        else:
            raise Exception("Unrecognized staining method.")

        self.stain_matrix_target = None
        self.target_concentration = None
        self.maxC_target = None
        self.stain_matrix_target_RGB = None

        self.method = method.lower()

    def fit(self, target):
        """
        Fi to target image: estimate the reference stain.

        :param target: Image RGB uint8.
        :return:
        """
        self.stain_matrix_target = self.extractor.get_stain_matrix(target)
        self.target_concentration = self.extractor.get_concentrations(target, self.stain_matrix_target)
        self.maxC_target = np.percentile(self.target_concentration, 99, axis=0).reshape((1, 2))
        self.stain_matrix_target_RGB = convert_OD_to_RGB(self.stain_matrix_target)  # for visualization.

    def transform(self, I):
        """
        Transform an image.
        :param I: Image RGB uint8.
        :return: Image RGB uint8 with the stain normalized using the target stain.
        """
        assert self.stain_matrix_target is not None and self.maxC_target is not None, "Fit first a target."

        stain_matrix_source = self.extractor.get_stain_matrix(I)
        source_concentrations = self.extractor.get_concentrations(I, stain_matrix_source)
        maxC_source = np.percentile(source_concentrations, 99, axis=0).reshape((1, 2))
        source_concentrations *= (self.maxC_target / maxC_source)

        tmp = 255 * np.exp(-1 * np.dot(source_concentrations, self.stain_matrix_target))
        return tmp.reshape(I.shape).astype(np.uint8)

    def __call__(self, I):
        """
        Transform the image I using a fitted stain matrix.
        The same as self.transform(), but it returns the normalized image, its concentrations matrix, and its tissue
        mask.

        :param I: Image RGB uint8 (h, w, 3).
        :return: Image RGB uint8 (the transformed image), 2D float64 concentrations matrix of size (h, w, nbr_stains),
                2D binary mask of the tissue.

        """
        assert self.stain_matrix_target is not None and self.maxC_target is not None, "You need to fit a target first."

        h, w, c = I.shape
        tissue_mask = get_tissue_mask(I)

        stain_matrix_source = self.extractor.get_stain_matrix(I)
        source_concentrations = self.extractor.get_concentrations(I, stain_matrix_source)
        maxC_source = np.percentile(source_concentrations, 99, axis=0).reshape((1, 2))
        source_concentrations *= (self.maxC_target / maxC_source)

        tmp = 255 * np.exp(-1 * np.dot(source_concentrations, self.stain_matrix_target))
        return tmp.reshape(I.shape).astype(np.uint8), source_concentrations.reshape((h, w, -1)), tissue_mask

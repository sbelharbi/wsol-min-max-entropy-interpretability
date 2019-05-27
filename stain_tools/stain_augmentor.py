# Main references:
# https://github.com/Peter554/StainTools
import numpy as np
import copy

from stain_tools.stain_extractor import MacenkoStainExtractor, VahadaneStainExtractor
from stain_tools.tools import get_tissue_mask

__all__ = ["StainAugmentor"]


class StainAugmentor(object):
    """
    Augment an image: generate another image slightly similar (useful for data augmentation when training).
    The augmentation is based on adding small uniform noise to the concentration matrix of the input image.
    """
    def __init__(self, method, sigma1=0.2, sigma2=0.2, augment_background=True):
        if method.lower() == "macenko":
            self.extractor = MacenkoStainExtractor
        elif method.lower() == "vahadane":
            self.extractor = VahadaneStainExtractor
        else:
            raise Exception("Unrecognized staining method.")

        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.augment_background = augment_background
        self.stain_matrix = None

    def set_stain_matrix(self, stain_matrix):
        """
        Set the stain matrix to some predefined matrix.
        :param stain_matrix: 2D numpy array of float64 of size (nbr_stains, nbr_plans). For instance, in the case of
               H&E staining, nbr_stains = 2, and if the images are RGB, nbr_plans = 3.
        :return:
        """
        self.stain_matrix = copy.deepcopy(stain_matrix)
        print("The target stain matrix has just been set using some matrix .... [OK]")

    def fit(self, I):
        """
        Fit an image.

        :param I: Input image RGB uint8.
        :return:
        """
        self.image_shape = I.shape
        self.stain_matrix = self.extractor.get_stain_matrix(I)
        self.source_concentrations = self.extractor.get_concentrations(I, self.stain_matrix)
        self.n_stains = self.source_concentrations.shape[1]
        self.tissue_mask = get_tissue_mask(I).ravel()

    def pop(self):
        """
        Get an augmented version of the fitted image.
        :return: RGB uint8 image.
        """
        augmented_concentrations = copy.deepcopy(self.source_concentrations)

        for i in range(self.n_stains):
            alpha = np.random.uniform(1 - self.sigma1, 1 + self.sigma1)
            beta = np.random.uniform(- self.sigma2, self.sigma2)
            if self.augment_background:
                augmented_concentrations[:, i] *= alpha
                augmented_concentrations[:, i] += beta
            else:
                augmented_concentrations[self.tissue_mask, i] *= alpha
                augmented_concentrations[self.tissue_mask, i] += beta

        I_augmented = 255 * np.exp(-1 * np.dot(augmented_concentrations, self.stain_matrix))
        I_augmented = I_augmented.reshape(self.image_shape)
        I_augmented = np.clip(I_augmented, 0, 255)

        return I_augmented

    def augment(self, I):
        """
        Perform fit(I), and pop() at once. Can be useful when multi-threading to avoid any issue of sharing self.mtx
        :param I: Input RGB uint8 image.
        :return: Randomly augmented image, 2D
        """
        # Fit
        image_shape = I.shape
        stain_matrix = self.extractor.get_stain_matrix(I)
        source_concentrations = self.extractor.get_concentrations(I, stain_matrix)
        n_stains = source_concentrations.shape[1]
        tissue_mask = get_tissue_mask(I).ravel()

        # Augment
        augmented_concentrations = copy.deepcopy(source_concentrations)

        for i in range(n_stains):
            alpha = np.random.uniform(1 - self.sigma1, 1 + self.sigma1)
            beta = np.random.uniform(- self.sigma2, self.sigma2)
            if self.augment_background:
                augmented_concentrations[:, i] *= alpha
                augmented_concentrations[:, i] += beta
            else:
                augmented_concentrations[tissue_mask, i] *= alpha
                augmented_concentrations[tissue_mask, i] += beta

        I_augmented = 255 * np.exp(-1 * np.dot(augmented_concentrations, stain_matrix))
        I_augmented = I_augmented.reshape(image_shape)
        I_augmented = np.clip(I_augmented, 0, 255)

        return I_augmented

    def __call__(self, source_concentrations, tissue_mask):
        """
        Augment an image based on its concentrations matrix and its tissue mask.

        :param source_concentrations: 2D numpy array of size (h*w, nbr_stains) of float64.
        :param tissue_mask: Binary tissue mask, a numpy vector of shape (h, w).
        :return: RGB uint8 2D image augmented.
        """
        assert self.stain_matrix is not None, "stain_matrix is None .... [NOT OK]"
        assert isinstance(source_concentrations, np.ndarray), "Concentration matrix must be `{}`. Its type is `{}` " \
                                                              ".... [NOT OK]".format(np.ndarray,
                                                                                     type(source_concentrations))
        assert source_concentrations.ndim == 2, "Concentrations matrix must be a 2D matrix. It is `{}` .... [NOT " \
                                                "OK]".format(source_concentrations.ndim)
        assert source_concentrations.dtype == np.float64, "Concentration matrix data must be of type `{}`. " \
                                                          "It is `{}` .... [NOT OK]".format(
                                                           np.float64, source_concentrations.dtype)

        assert isinstance(tissue_mask, np.ndarray), "Tissue mask must be of type `{}`. " \
                                                    "It is `{}` .... [NOT OK]".format(np.ndarray, type(tissue_mask))
        assert tissue_mask.ndim == 2, "Tissue mask must be a 2D matrix. It is `{}`  .... [NOT OK]".format(
            tissue_mask.ndim)
        assert tissue_mask.dtype == np.bool, "Tissue mask data must be of type `{}`. It is `{}` .... [NOT OK]".format(
            np.bool, tissue_mask.dtype
        )

        h, w = tissue_mask.shape
        n_plans = self.stain_matrix.shape[1]
        image_shape = (h, w, n_plans)
        n_stains = source_concentrations.shape[1]
        tissue_mask = tissue_mask.ravel()

        # Augment
        augmented_concentrations = copy.deepcopy(source_concentrations)

        for i in range(n_stains):
            alpha = np.random.uniform(1 - self.sigma1, 1 + self.sigma1)
            beta = np.random.uniform(- self.sigma2, self.sigma2)
            if self.augment_background:
                augmented_concentrations[:, i] *= alpha
                augmented_concentrations[:, i] += beta
            else:
                augmented_concentrations[tissue_mask, i] *= alpha
                augmented_concentrations[tissue_mask, i] += beta

        I_augmented = 255 * np.exp(-1 * np.dot(augmented_concentrations, self.stain_matrix))
        I_augmented = I_augmented.reshape(image_shape)
        I_augmented = np.clip(I_augmented, 0, 255)

        return I_augmented.astype(np.uint8)

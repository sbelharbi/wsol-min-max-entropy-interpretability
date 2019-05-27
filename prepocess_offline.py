import collections
import os
import pickle as pkl
import datetime as dt

from PIL import Image
import numpy as np
import tqdm

import stain_tools.stain_normalizer as stain_normalizer
import stain_tools.tools as stain_tools


__all__ = ["Preprocessor"]


class Preprocessor(object):
    """
    Preprocess a data set, and save it on disc.
    This function is time consuming. It is not even considerable to use it during training.
    It is better to do it offline and save everything on disc. Then load it when needed.
    It was designed to work offline by calling it on an entire dataset.

    The preprocessing consists in standardizing the brightness of the images (H&E) and normalize their stain.
    The images are resize first, if requested.

    The output will be saved in a folder.

    For each image, we store the following information as a dictionary:
        "label": label,  # the label of the image.
        "img_array": img_array,  # the image as numpy.ndarray of RGB uint8.
        "concent": concent,  # the concentration matrix of the image.
        "tissue_mask": tissue_mask,  # the tissue make of the image.
        "method": method,  # the method name of the stain normalizer.
        "stain_matrix_target": stain_matrix_target  # the reference stain matrix.
        "absolute_path": the absolute path to the image. (can be useful for a quick debug).
    """
    def __init__(self, stain, name_classes):
        """
        Init function.
        :param stain: object, defines the operations to do. Attributes:
                      brightnessStandardizer: str, (or None), the name of the class that standardizes the brightness.
                      percentile: positive float in ]0, 100], a percentile (related to brightnessStandardizer class).
                      stainNormalizer: str, (or None), the name of the class that normalizes the stain.
                      method: str, name of the stain normalization method (for the class of stainNormalizer).
                      target: str, path to the target image for stain normalization (for the class of stainNormalizer).
                      resize: positive int, or tuple (W, H)!!!!!!!!!!!!! of positive int, or None. The size of the
                      image to which it will be resized to before doing any preprocessing on the images themselves.
        :param name_classes: dict, {"classe_name": int}.
        """
        super(Preprocessor, self).__init__()

        self.stain = stain
        self.brightnessStandardizer = None
        self.stainNormalizer = None
        self.resize = None

        self.name_classes = name_classes

        if stain.brightnessStandardizer:
            self.brightnessStandardizer = stain_tools.__dict__[stain.brightnessStandardizer](
                percentile=stain.percentile
            )

        if stain.stainNormalizer:
            self.stainNormalizer = stain_normalizer.__dict__[stain.stainNormalizer](stain.method)

        if self.stainNormalizer:
            self.imageTarget = Image.open(stain.target, 'r').convert("RGB")
            if stain.resize:
                if isinstance(stain.resize, int):
                    self.resize = (stain.resize, stain.resize)
                elif isinstance(stain.resize, collections.Sequence):
                    self.resize = stain.resize

                self.imageTarget = self.imageTarget.resize(self.resize)
                print("Target' size has been normalized into" + str(self.resize) + " .... [OK]")

            self.imageTarget = Image.fromarray(
                self.brightnessStandardizer(np.asarray(self.imageTarget)), mode="RGB"
            )  # normalize the brightness.
            print("Target's brightness has been standardized .... [OK]")
            self.stainNormalizer.fit(np.asarray(self.imageTarget).copy())  # extract the target stain.
            print("Target's stain matrix has been extracted .... [OK]")

    def __call__(self, data, outd, name):
        """
        Preprocess a specific data.

        :param data: list of absolute paths (str) to images.
        :param outd: str, absolute path to the output directory.
        :param name: str, base name of the file (no extension).
        :return:
        """
        t0 = dt.datetime.now()
        print("Start prepcessing {} files  .... [OK]".format(name))
        samples = []
        n = len(data)
        # n = min(len(data), 5)
        for i, s in tqdm.tqdm(enumerate(data[:n]), ncols=80, total=n):
            samples.append(self.preprocess_sample(s))

        print(
            "Finished preprocessing. Going to dump the results in the disc. It may take some time, but it is "
            "supposed to be fast .... [OK]"
        )

        self.save_data_into_numpy_format(samples, outd, name)

        dest = os.path.join(outd, name)

        output_mssg = "{} has preprocessed the data, and " \
                      "saved the results in {} in time of {} .... [OK].".format(self.__class__.__name__, dest,
                                                                                str(dt.datetime.now() - t0))
        print(output_mssg)

        # Save to thumbnails for checking.
        self.save_thumbnail(samples, outd, name)
        output_mssg = "{} has saved thumbnails of the preprocessed data " \
                      "in {} .... [OK].".format(self.__class__.__name__, os.path.join(outd, name))
        print(output_mssg)

        # Save the stain target path.
        with open(os.path.join(outd, "stain-target-absolute-path.txt"), 'w') as fx:
            fx.write("Stain target absolute path: " + str(self.stain.target))

    def save_data_into_numpy_format(self, samples, outd, name):
        """
        Fast save (in order to Fast load) the data.
        We use numpy.

        How data is saved?
        Every sample has a set of information:
            label: int, the class label.
            img_array: 2D RGB uint8 image (h, w, 3) (numpy array).
            concent: 2D float64 concentrations matrix of size (h, w, nbr_stains) (numpy array).
            tissue_mask: 2D binary mask of the tissue of size (h, w) (numpy array).
            method: str, name of the stain normalization method.
            absolute_path: str, absolute path to the image.

        Everything will be saved into the FOLDER: outd/name/
        Files to be stored are the following:
            metadata.pkl: contains a list of dict. Each dict is for a sample and it has the following keys with str
            values:
                label
                method
                absolute_path
            img_array_i.npz: stores the 'img_array' for the ith sample.
            concent_i.npz: stores 'concent' for the ith sample/
            tissue_mask.npz: stores 'tissue_mask' for the ith sample.

        :param samples: list of samples (the result within self.__call__()).
        :param outd: str, absolute path to the output directory.
        :param name: str, base name of the file (no extension).
        :return:
        """
        OUTD = os.path.join(outd, name)
        if not os.path.exists(OUTD):
            os.makedirs(OUTD)

        metadata = []
        n = len(samples)
        for i, sample in tqdm.tqdm(enumerate(samples), ncols=80, total=n):
            # =================== Textual info: metadata. =========================
            metadata.append(
                {"label": sample["label"],
                 "method": sample["method"],
                 "absolute_path": sample["absolute_path"]
                 }
            )

            # =================== numpy.ndarray data.   ===========================
            img_array_ = sample["img_array"]
            concent_ = sample["concent"]
            tissue_mask_ = sample["tissue_mask"]
            stain_matrix_target_ = sample["stain_matrix_target"]

            # save numpy.ndarray data into *.npz
            np.savez_compressed(os.path.join(OUTD, "img_array_" + str(i)), img_array=img_array_)
            np.savez_compressed(os.path.join(OUTD, "concent_" + str(i)), concent=concent_)
            np.savez_compressed(os.path.join(OUTD, "tissue_mask_" + str(i)), tissue_mask=tissue_mask_)
            np.savez_compressed(
                os.path.join(OUTD, "stain_matrix_target_" + str(i)), stain_matrix_target=stain_matrix_target_
            )

        # save the meta data into pkl
        with open(os.path.join(OUTD, "metadata.pkl"), "wb") as fout:
            pkl.dump(metadata, fout, protocol=pkl.HIGHEST_PROTOCOL)

    def preprocess_sample(self, s):
        """
        Preprocess a sample.

        :param s: str, absolute path to the image.
        :return: dict, with the keys:
                label: int, the class label.
                img_array: 2D RGB uint8 image (h, w, 3) (numpy array).
                concent: 2D float64 concentrations matrix of size (h, w, nbr_stains) (numpy array).
                tissue_mask: 2D binary mask of the tissue of size (h, w) (numpy array).
                method: str, name of the stain normalization method.
                absolute_path: str, absolute path to the image.
        """
        img = Image.open(s, "r").convert("RGB")
        label = self.name_classes[s.split(os.sep)[-2]]
        if self.resize:
            img = img.resize(self.resize)

        img_array = np.asarray(img)

        if self.brightnessStandardizer:
            img_array = self.brightnessStandardizer(img_array)

        concent, tissue_mask, method, stain_matrix_target = None, None, None, None

        if self.stainNormalizer:
            img_array, concent, tissue_mask = self.stainNormalizer(img_array)
            method = self.stainNormalizer.method
            stain_matrix_target = self.stainNormalizer.stain_matrix_target

        return {"label": label,
                "img_array": img_array,
                "concent": concent,
                "tissue_mask": tissue_mask,
                "method": method,
                "stain_matrix_target": stain_matrix_target,
                "absolute_path": s}

    def save_thumbnail(self, samples, outd, name):
        """
        Save thumbnail of the original images in low resolution. (with the same size).
        This can be helpful to check the images after the preprocessing. For instance, in case, one selects a random
        target, it is better to double check that the normalized images are correct.

        :param samples: List, of samples returned by self.preprocess_sample().
        :param outd: str, absolute path to the output directory. (see self.__call__())
        :param name: str, base name of the file (no extension). (see self.__call__())
        :return:
        """
        OUTD = os.path.join(outd, name + "-thumbnails")
        if not os.path.exists(OUTD):
            os.makedirs(OUTD)

        n = len(samples)
        for i, s in tqdm.tqdm(enumerate(samples), ncols=80, total=n):
            file_name = s["absolute_path"].split(os.sep)[-2] + '-' + s["absolute_path"].split(os.sep)[-1].split(".")[0]
            im = Image.fromarray(s["img_array"], mode="RGB")
            im.thumbnail(im.size)
            im.save(os.path.join(OUTD, file_name + ".jpeg"), "JPEG")



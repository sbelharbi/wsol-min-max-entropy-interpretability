import datetime as dt
from os.path import join

import yaml

# TODO: Adjust the exp. generator.

# DEFAULT configuration.
config = {
    # ######################### GENERAL STUFF ##############################
    "MYSEED": 0,  # Seed for reproducibility. [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, ...]
    "dataset": "Caltech-UCSD-Birds-200-2011",  # name of the dataset: glas, Caltech-UCSD-Birds-200-2011
    "img_extension": "bmp",  # extension of the images in the dataset.
    "name_classes": "encoding.yaml",  # {'benign': 0, 'malignant': 1},  # dict. name classes and corresponding int. If
    # dict if too big,
    # you can dump it in the fold folder in a yaml file. We will load it when needed. Use the name of the file.
    "nbr_classes": 200,  # Total number of classes. glas: 2, Caltech-UCSD-Birds-200-2011: 200.
    "split": 0,  # split id.
    "fold": 0,  # folder id.
    "fold_folder": "./folds",  # relative path to the folder of the folds.
    "resize": None,  # PIL format of the image size (w, h). The size to which the original images are resized to.
    "crop_size": (480, 480),  # Size of the patches to be cropped (h, w).
    "up_scale_small_dim_to": 500,  # None # int or None. If int, the images are upscaled to this size while
    # preserving the ratio. See loader.PhotoDataset().
    "nbr_times_erase": 0,  # number of time to erase discriminative regions during TRAINING. 0: None.
    "epoch_start_erasing": 1,  # epoch when we start erasing.
    "sigma_erase": 10,  # std deviation tha allows to shape the slop of the exp(-t/sigma).
    "erase_in_inference": False,  # If True, we perform erasing during inference.
    "padding_size": None,  # (0.5, 0.5),  # padding ratios for the original image for (top/bottom) and (left/right).
    # Can be
    # applied on both, training/evaluation modes. To be specified in PhotoDataset(). If specified, only training
    # images are padded. To pad evaluation images, you need to set the variable: `pad_eval` to True.
    "pad_eval": False,  # If True, evaluation images are padded in the same way. The final mask is cropped inside the
    # predicted mask (since this last one is bigger due to the padding).
    "padding_mode": "reflect",  # type of padding. Accepted modes:
    # https://pytorch.org/docs/stable/torchvision/transforms.html#torchvision.transforms.functional.pad
    "preload": True,  # If True, images are loaded and saved in RAM to avoid disc access.
    "batch_size": 8,  # the batch size for training.
    "num_workers": 8,  # number of workers for dataloader of the trainset.
    "max_epochs": 400,  # number of training epochs.
    # ############################# VISUALISATION OF REGIONS OF INTEREST ####################
    "normalize": True,  # If True, maps are normalized using softmax. [NOT USED IN THIS CODE]
    "alpha": 128,  # transparency alpha, used for plotting. In [0, 255]. The lower the value, the more transparent
    # the map.
    "floating": 3,  # the number of floating points to print over the maps.
    "height_tag": 50,  # the height of the margin where the tag is written.
    "use_tags": True,  # If True, extra information will be display under the images.
    "show_hists": True,  # If True, histograms of scores will be displayed as density probability.
    "bins": 100,  # int, number of bins in the histogram.
    "rangeh": (0, 1),  # tuple, range of the histogram.
    "extension": ("jpeg", "JPEG"),  # format into which the maps are saved.
    # ######################### Optimizer ##############################
    "optimizer": {  # the optimizer
        # ==================== SGD =======================
        "name": "sgd",  # str name.
        "lr": 0.001,  # Initial learning rate.
        "momentum": 0.9,  # Momentum.
        "dampening": 0.,  # dampening.
        "weight_decay": 1e-5,  # The weight decay (L2) over the parameters.
        "nesterov": True,  # If True, Nesterov algorithm is used.
        # ==================== ADAM =========================
        # "name": "adam",  # str name.
        # "lr": 0.0001,  # Initial learning rate.
        # "betas": (0.9, 0.999),  # betas.
        # "weight_decay": 0.0005,  # The weight decay (L2) over the parameters.
        # "eps": 1e-08,  # eps. for numerical stability.
        # "amsgrad": False,  # Use amsgrad variant or not.
        # ========== LR scheduler: how to adjust the learning rate. =========================
        "lr_scheduler": {
            # ========> torch.optim.lr_scheduler.StepLR
            # "name": "step",  # str name.
            # "step_size": 20,  # Frequency of which to adjust the lr.
            # "gamma": 0.1,  # the update coefficient: lr = gamma * lr.
            # "last_epoch": -1,  # the index of the last epoch where to stop adjusting the LR.
            # ========> MyStepLR: override torch.optim.lr_scheduler.StepLR
            "name": "mystep",  # str name.
            "step_size": 40,  # Frequency of which to adjust the lr.
            "gamma": 0.1,  # the update coefficient: lr = gamma * lr.
            "last_epoch": -1,  # the index of the last epoch where to stop adjusting the LR.
            "min_lr": 1e-7,  # minimum allowed value for lr.
            # ========> torch.optim.lr_scheduler.MultiStepLR
            # "name": "multistep",  # str name.
            # "milestones": [0, 100],  # milestones.
            # "gamma": 0.1,  # the update coefficient: lr = gamma * lr.
            # "last_epoch": -1  # the index of the last epoch where to stop adjusting the LR.
        }
    },
    # ######################### Model ##############################
    "model": {
        "name": "resnet101",  # name of the classifier.
        "pretrained": True,  # use/or not the ImageNet pretrained models.
        # =============================  classifier ==========================
        "num_classes": 200,  # number of output classes. glas: 2, Caltech-UCSD-Birds-200-2011: 200.
        "scale": (0.5, 0.5),  # ratio used to scale the input images for the classifier.
        "modalities": 5,  # number of modalities (wildcat).
        "kmax": 0.1,  # kmax. (wildcat)
        "kmin": 0.1,  # kmin. (wildcat)
        "alpha": 0.0,  # alpha. (wildcat)
        "dropout": 0.0,  # dropout over the kmin and kmax selected activations.. (wildcat).
        # ===============================  Segmentor ===========================
        "num_masks": 1,  # number of the masks. (only 1 is accepted)
        "sigma": 0.5,  # simga for the thresholding.
        "w": 8  # w for the thresholding.
    }
}


fold_yaml = "config_yaml"
fold_bash = "config_bash"
name_config = dt.datetime.now().strftime('%m_%d_%Y_%H_%M_%S_%f')
name_bash = join(fold_bash, name_config + ".sh")
name_yaml = join(fold_yaml, name_config + ".yaml")

with open(name_yaml, 'w') as f:
    yaml.dump(config, f)

cmd = "time python train_deepmil.py --yaml " + name_config + ".yaml" + " --cudaid 0"
with open(name_bash, 'w') as f:
    f.write("#!/usr/bin/env bash \n")
    f.write("export MYSEED={}\n".format(config["MYSEED"]))
    f.write(cmd + "\n")

# TODO:
#  1. Create slurm host-jobs 2. Generate many configs..



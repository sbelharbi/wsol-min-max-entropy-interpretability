from collections import Sequence
import warnings

from torch.optim import SGD
from torch.optim import Adam
import torch.optim.lr_scheduler as lr_scheduler


from deepmil import models, criteria
from deepmil import lr_scheduler as my_lr_scheduler
from tools import Dict2Obj, count_nb_params
import loader

import prepocess_offline
import stain_tools.stain_augmentor as stain_augmentors


def instantiate_train_loss(args):
    """
    Instantiate the train loss.

    :param args: object. Contains the configuration of the exp that has been read from the yaml file.
    :return: train_loss: instance of deepmil.criteria.TotalLoss()
    """
    return criteria.TotalLoss()


def instantiate_eval_loss(args):
    """
    Instantiate the evaluation (test phase) loss.

    :param args: object. Contains the configuration of the exp that has been read from the yaml file.
    :return: eval_loss: instance of deepmil.criteria.TotalLossEval()
    """
    return criteria.TotalLossEval()


def instantiate_models(args):
    """Instantiate the necessary models.
    Input:
        args: object. Contains the configuration of the exp that has been read from the yaml file.

    Output:
        segmentor: instance of module from deepmil.representation; Embeds the instance.
        classifier: instance of module from deepmil.decision_pooling; pools the score of each class.
    """
    p = Dict2Obj(args.model)
    model = models.__dict__[p.name](pretrained=p.pretrained, num_masks=p.num_masks,
                                    sigma=p.sigma, w=p.w, num_classes=p.num_classes, scale=p.scale,
                                    modalities=p.modalities, kmax=p.kmax, kmin=p.kmin, alpha=p.alpha,
                                    dropout=p.dropout, nbr_times_erase=args.nbr_times_erase,
                                    sigma_erase=args.sigma_erase)

    print("Mi-max entropy model `{}` was successfully instantiated. Nbr.params: {} .... [OK]".format(
        model.__class__.__name__, count_nb_params(model)))
    return model


def instantiate_optimizer(args, model):
    """Instantiate an optimizer.
    Input:
        args: object. Contains the configuration of the exp that has been read from the yaml file.
        mode: a pytorch model with parameters.

    Output:
        optimizer: a pytorch optimizer.
        lrate_scheduler: a pytorch learning rate scheduler (or None).
    """
    if args.optimizer["name"] == "sgd":
        optimizer = SGD(model.parameters(), lr=args.optimizer["lr"], momentum=args.optimizer["momentum"],
                        dampening=args.optimizer["dampening"], weight_decay=args.optimizer["weight_decay"],
                        nesterov=args.optimizer["nesterov"])
    elif args.optimizer["name"] == "adam":
        optimizer = Adam(params=model.parameters(), lr=args.optimizer["lr"], betas=args.optimizer["betas"],
                         eps=args.optimizer["eps"], weight_decay=args.optimizer["weight_decay"],
                         amsgrad=args.optimizer["amsgrad"])
    else:
        raise ValueError("Unsupported optimizer `{}` .... [NOT OK]".format(args.optimizer["name"]))

    print("Optimizer `{}` was successfully instantiated .... [OK]".format([key + ":" + str(args.optimizer[key]) for
                                                                           key in args.optimizer.keys()]))

    if args.optimizer["lr_scheduler"]:
        if args.optimizer["lr_scheduler"]["name"] == "step":
            lr_scheduler_ = args.optimizer["lr_scheduler"]
            lrate_scheduler = lr_scheduler.StepLR(optimizer,
                                                  step_size=lr_scheduler_["step_size"],
                                                  gamma=lr_scheduler_["gamma"],
                                                  last_epoch=lr_scheduler_["last_epoch"])
            print("Learning scheduler `{}` was successfully instantiated .... [OK]".format(
                [key + ":" + str(lr_scheduler_[key]) for key in lr_scheduler_.keys()]
            ))
        elif args.optimizer["lr_scheduler"]["name"] == "mystep":
            lr_scheduler_ = args.optimizer["lr_scheduler"]
            lrate_scheduler = my_lr_scheduler.MyStepLR(optimizer,
                                                       step_size=lr_scheduler_["step_size"],
                                                       gamma=lr_scheduler_["gamma"],
                                                       last_epoch=lr_scheduler_["last_epoch"],
                                                       min_lr=lr_scheduler_["min_lr"])
            print("Learning scheduler `{}` was successfully instantiated .... [OK]".format(
                [key + ":" + str(lr_scheduler_[key]) for key in lr_scheduler_.keys()]
            ))
        elif args.optimizer["lr_scheduler"]["name"] == "multistep":
            lr_scheduler_ = args.optimizer["lr_scheduler"]
            lrate_scheduler = lr_scheduler.MultiStepLR(optimizer,
                                                       milestones=lr_scheduler_["milestones"],
                                                       gamma=lr_scheduler_["gamma"],
                                                       last_epoch=lr_scheduler_["last_epoch"])
            print("Learning scheduler `{}` was successfully instantiated .... [OK]".format(
                [key + ":" + str(lr_scheduler_[key]) for key in lr_scheduler_.keys()]
            ))
        else:
            raise ValueError("Unsupported learning rate scheduler `{}` .... [NOT OK]".format(
                args.optimizer["lr_scheduler"]["name"]))
    else:
        lrate_scheduler = None

    return optimizer, lrate_scheduler


def instantiate_preprocessor(args):
    """
    Instantiate a preprocessor class from preprocess_offline.

    :param args: object. Contains the configuration of the exp that has been read from the yaml file.
    :return: an instance of a preprocessor.
    """
    if args.preprocessor:
        if args.preprocessor["name"] == "Preprocessor":
            if "stain" in args.preprocessor.keys():
                stain = Dict2Obj(args.preprocessor["stain"])
                name_classes = args.name_classes
                preprocessor = prepocess_offline.__dict__["Preprocessor"](stain, name_classes)

                print(
                    "Preprocessor `{}` was successfully instantiated with the stain preprocessing ON .... [OK]".format(
                        args.preprocessor["name"])
                )

                return preprocessor
            else:
                raise ValueError("Unknown preprocessing operation .... [NOT OK]")
        else:
            raise ValueError("Unsupported preprocessor `{}` .... [NOT OK]".format(args.preprocessor["name"]))
    else:
        print("Proceeding WITHOUT preprocessor .... [OK]")
        return None


def instantiate_patch_splitter(args, deterministic=True):
    """
    Instantiate the patch splitter and its relevant instances.

    For every set.
    However, for train, determninistic is set to False to allow dropout over the patches IF requiested.
    Over valid an test sets, deterministic is True.

    :param args: object. Contains the configuration of the exp that has been read from the yaml file.
    :param deterministic: True/False. If True, dropping some samples will be allowed IF it was requested. Should set
           to True only with the train set.
    :return: an instance of a patch splitter.
    """
    assert args.patch_splitter is not None, "We need a patch splitter, and you didn't specify one! .... [NOT OK]"
    patch_splitter_conf = Dict2Obj(args.patch_splitter)
    random_cropper = Dict2Obj(args.random_cropper)
    if patch_splitter_conf.name == "PatchSplitter":
        keep = 1.  # default value for deterministic scenario: keep all patch (evaluation phase).
        if not deterministic:
            keep = patch_splitter_conf.keep

        h = patch_splitter_conf.h
        w = patch_splitter_conf.w
        h_ = patch_splitter_conf.h_
        w_ = patch_splitter_conf.w_

        # Instantiate the patch transforms if there is any.
        patch_transform = None
        if patch_splitter_conf.patch_transform:
            error_msg = "We support only one or none patch transform for now ... [NOT OK]"
            assert not isinstance(patch_splitter_conf.patch_transform, Sequence), error_msg

            patch_transform_config = Dict2Obj(patch_splitter_conf.patch_transform)
            if patch_transform_config.name == "PseudoFoveation":
                scale_factor = patch_transform_config.scale_factor
                int_eps = patch_transform_config.int_eps
                num_workers = patch_transform_config.num_workers

                patch_transform = loader.__dict__["PseudoFoveation"](h, w, h_, w_, scale_factor, int_eps, num_workers)

                print(
                    "Patch transform `{}` was successfully instantiated WITHIN a patch splitter `{}`"
                    "with `{}` workers.... [OK]".format(
                        patch_transform_config.name, patch_splitter_conf.name, num_workers)
                )

            elif patch_transform_config.name == "FastApproximationPseudoFoveation":
                scale_factor = patch_transform_config.scale_factor
                int_eps = patch_transform_config.int_eps
                nbr_kernels = patch_transform_config.nbr_kernels
                use_gpu = patch_transform_config.use_gpu
                gpu_id = patch_transform_config.gpu_id

                if gpu_id is None:
                    gpu_id = int(args.cudaid)
                    warnings.warn("You didn't specify the CUDA device ID to run `FastApproximationPseudoFoveation`. "
                                  "We set it up to the same device where the model will be run `cuda:{}` .... [NOT "
                                  "OK]".format(args.cudaid))

                assert args.num_workers in [0, 1], "'config.num_workers' must be in {0, " \
                                                   "1} if loader.FastApproximationPseudoFoveation() is used. " \
                                                   "Multiprocessing does not play well when Dataloader has uses also " \
                                                   "multiprocessing .... [NOT OK]"

                patch_transform = loader.__dict__["FastApproximationPseudoFoveation"](
                    h, w, h_, w_, scale_factor, int_eps, nbr_kernels, use_gpu, gpu_id
                )

                print(
                    "Patch transform `{}` was successfully instantiated WITHIN a patch splitter `{}` "
                    "with `{}` kernels with `{}` GPU and CUDA ID `{}` .... [OK]".format(
                        patch_transform_config.name, patch_splitter_conf.name, nbr_kernels, use_gpu, gpu_id)
                )

            else:
                raise ValueError("Unsupported patch transform `{}`  .... [NOT OK]".format(patch_transform_config.name))
        else:
            print("Proceeding WITHOUT any patch transform  ..... [OK]")

        if patch_transform:
            patch_transform = [patch_transform]

        padding_mode = patch_splitter_conf.padding_mode
        assert hasattr(random_cropper, "make_cropped_perfect_for_split"), "The random cropper `{}` does not have the " \
                                                                          "attribute `make_cropped_perfect_for_split`" \
                                                                          "which we expect .... [NO OK]".format(
                                                                          random_cropper.name)
        if random_cropper.make_cropped_perfect_for_split and not deterministic:
            padding_mode = None
        patch_splitter = loader.__dict__["PatchSplitter"](
            h, w, h_, w_, padding_mode, patch_transforms=patch_transform, keep=keep
        )

        print("Patch splitter `{}` was successfully instantiated .... [OK]".format(patch_splitter_conf.name))

    else:
        raise ValueError("Unsupported patch splitter `{}` .... [NOT OK]".format(patch_splitter_conf.name))

    return patch_splitter


def instantiate_stain_augmentor(args):
    """
    Instantiate the stain augmentor.
    The possible classes are located in stain_tools.stain_augmentor.

    :param args: object. Contains the configuration of the exp that has been read from the yaml file.
    :return: an instance of stain augmentor, or None.
    """
    if args.stain_augmentor:
        error_msg = "You requested stain augmentation, but there was no stain normalization. It seems inconsistent." \
                    "Modify the code in order to accept a stain augmentation without stain normalization. Stain " \
                    "extraction is time consuming. To augment the stains, we use the same reference stain in the" \
                    "stain normalization phase. If you want to stain augmentation anyway, you need to provide a" \
                    "stain matrix because stain extration takes about 15 to 25 seconds per H&E high image of size" \
                    "hxw: ~1500x2000."
        assert "stain" in args.preprocessor.keys(), error_msg

        method = args.preprocessor["stain"]["method"]

        s_augmentor_config = Dict2Obj(args.stain_augmentor)

        if s_augmentor_config.name == "StainAugmentor":
            sigma1 = s_augmentor_config.sigma1
            sigma2 = s_augmentor_config.sigma2
            augment_background = s_augmentor_config.augment_background

            stain_augmentor = stain_augmentors.__dict__["StainAugmentor"](method, sigma1, sigma2, augment_background)

            print("Stain augmentor `{}` was successfully instantiated .... [OK]".format(s_augmentor_config.name))

            return stain_augmentor
        else:
            raise ValueError("Unsupported stain augmentor name `{}` .... [NOT OK]".format(s_augmentor_config.name))
    else:
        print("Proceeding WITHOUT stain augmentation .... [OK]")
        return None


def instantiante_random_cropper(args):
    """
    Instantiate a random cropper. It is used for sampling su-images from an original image in the train set.

    Classes are located in loader.*

    :param args: object. Contains the configuration of the exp that has been read from the yaml file.
    :return: an instance of a random cropper, or None.
    """
    if args.random_cropper:
        r_cropper_config = Dict2Obj(args.random_cropper)
        patch_splitter_config = Dict2Obj(args.patch_splitter)

        if r_cropper_config.name == "RandomCropper":
            min_height = r_cropper_config.min_height
            min_width = r_cropper_config.min_width
            max_height = r_cropper_config.max_height
            max_width = r_cropper_config.max_width
            make_cropped_perfect_for_split = r_cropper_config.make_cropped_perfect_for_split
            h, w, h_, w_ = None, None, None, None
            if make_cropped_perfect_for_split:
                assert patch_splitter_config.name == "PatchSplitter", "We expected the class `PatchSplitter`" \
                                                                      "but found `{}` .... [NOT OK]".format(
                                                                       patch_splitter_config.name)
                h = patch_splitter_config.h
                w = patch_splitter_config.w
                h_ = patch_splitter_config.h_
                w_ = patch_splitter_config.w_

            random_cropper = loader.__dict__["RandomCropper"](
                min_height, min_width, max_height, max_width, make_cropped_perfect_for_split, h, w, h_, w_)

            print("Random cropper `{}` was successfully instantiated .... [OK]".format(r_cropper_config.name))

            return random_cropper

        else:
            raise ValueError("Unsuppoerted random cropper `{}` .... [NOT OK]".format(r_cropper_config.name))
    else:
        return None



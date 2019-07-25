import argparse
import os
import numpy as np
from os.path import join
from copy import deepcopy
import datetime as dt
import warnings
import sys
import yaml
import random
import socket
import getpass
from collections import OrderedDict

import pickle as pkl

from torch.utils.data import DataLoader

from deepmil.train import train_one_epoch
from deepmil.train import validate

from tools import get_exp_name, copy_code, log, final_processing, load_pre_pretrained_model
from tools import get_device, get_rootpath_2_dataset, create_folders_for_exp
from tools import get_yaml_args, init_stats, plot_hist_probs_pos_neg, plot_roc_curve, get_cpu_device
from tools import get_transforms_tensor, get_train_transforms_img, plot_curves, announce_msg, superpose_curves
from tools import check_if_allow_multgpu_mode, copy_model_state_dict_from_gpu_to_cpu, get_state_dict

from loader import csv_loader, PhotoDataset, default_collate, MyDataParallel

from instantiators import instantiate_models, instantiate_optimizer, instantiate_train_loss, instantiate_eval_loss

import torch
import torch.nn as nn

import reproducibility


# TODO: check work on Glas: https://arxiv.org/pdf/1603.00275.pdf
FACTOR_MUL_WORKERS = 2  # args.num_workers * this_factor. Useful when setting set_for_eval to False, batch size =1,
# and we are in an evaluation mode (to go faster and coop with the lag between the CPU and GPU).
DEBUG_MODE = False  # Can be activated only for "Caltech-UCSD-Birds-200-2011" dataset to go fast. If True,
# we select only few samples for training, validation, and test.
PLOT_STATS = False


reproducibility.set_seed(None)  # use the default seed. Copy the see into the os.environ("MYSEED")

NBRGPUS = torch.cuda.device_count()

ALLOW_MULTIGPUS = check_if_allow_multgpu_mode()


def _init_fn(worker_id):
    """
    Init. function for the worker in dataloader.
    :param worker_id:
    :return:
    """
    pass
    # np.random.seed(int(os.environ["MYSEED"]))
    # random.seed(int(os.environ["MYSEED"]))
    # torch.manual_seed(int(os.environ["MYSEED"]))


if __name__ == "__main__":

    # =============================================
    # Parse the inputs and deal with the yaml file.
    # =============================================

    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", type=str, help="yaml file containing the configuration.")
    parser.add_argument("--cudaid", type=str, default="0", help="cuda id.")

    input_args, _ = parser.parse_known_args()

    args, args_dict = get_yaml_args(input_args)

    # ===============
    # Reproducibility
    # ===============

    # ==================================================
    # Device, criteria, folders, output logs, callbacks.
    # ==================================================

    DEVICE = get_device(args)
    CPUDEVICE = get_cpu_device()

    # TODO:
    #  In the case where the train loss is regularized, check if using the same loss for model selection is better than
    #  using only the plain cross-entropy.
    CRIT_TR = instantiate_train_loss(args).to(DEVICE)
    # TODO: use the same total loss. Fix totalloss to adjust automatically to the number of inputs.
    CRIT_EV = instantiate_eval_loss(args).to(DEVICE)

    FOLDER = '.'

    OUTD = join(FOLDER, "exps",
                # "nbr_erase-{}-".format(args.nbr_times_erase),
                "PID-{}-{}-bsz-{}-kmax-kmin-{}-dout-{}-erase-nbr-{}-at-epoch-{}-max-epochs-{}-stepsize-"
                "{}-nbr-modalitities-{}-lr-{}-mx-epochs-{}".format(
                    os.getpid(), get_exp_name(args), args.batch_size, args.model["kmax"], args.model["dropout"],
                    args.nbr_times_erase,
                    args.epoch_start_erasing, args.max_epochs, args.optimizer["lr_scheduler"]["step_size"],
                    args.model["modalities"], args.optimizer["lr"], args.max_epochs
                ))

    if not os.path.exists(OUTD):
        os.makedirs(OUTD)

    OUTD_TR = create_folders_for_exp(OUTD, "train")
    OUTD_VL = create_folders_for_exp(OUTD, "validation")
    OUTD_TS = create_folders_for_exp(OUTD, "test")

    subdirs = ["init_params"]
    for sbdr in subdirs:
        if not os.path.exists(join(OUTD, sbdr)):
            os.makedirs(join(OUTD, sbdr))

    # save the yaml file.
    if not os.path.exists(join(OUTD, "code/")):
        os.makedirs(join(OUTD, "code/"))
    with open(join(OUTD, "code/", input_args.yaml), 'w') as fyaml:
        yaml.dump(args_dict, fyaml)

    copy_code(join(OUTD, "code/"))

    training_log = join(OUTD, "training.txt")
    results_log = join(OUTD, "results.txt")

    log(training_log, "\n\n ########### Training #########\n\n")
    log(results_log, "\n\n ########### Results #########\n\n")
    # TODO: improve the logger.
    # check this https://stackoverflow.com/questions/4675728/redirect-stdout-to-a-file-in-python

    # TODO:
    #  check how Visdom/tensorboard works in Pytorch. Check also other visualization tools for future references such as
    #  this: https://medium.com/apache-mxnet/mxboard-mxnet-data-visualization-2eed6ae31d2c
    callback = None

    # ==========================================================
    # Data transformations: on PIL.Image.Image and torch.tensor.
    # ==========================================================

    train_transform_img = get_train_transforms_img(args)
    transform_tensor = get_transforms_tensor(args)

    # =======================================================================================================
    # Datasets: create folds, load csv, preprocess files and save on disc, load datasets: train, valid, test.
    # =======================================================================================================

    announce_msg("SPLIT: {} \t FOLD: {}".format(args.split, args.fold))

    relative_fold_path = join(
        args.fold_folder, args.dataset, "split_" + str(args.split), "fold_" + str(args.fold)
    )
    if isinstance(args.name_classes, str):  # path
        path_classes = join(relative_fold_path, args.name_classes)
        assert os.path.isfile(path_classes), "File {} does not exist .... [NOT OK]".format(path_classes)
        with open(path_classes, "r") as fin:
            args.name_classes = yaml.load(fin)

    train_csv = join(relative_fold_path, "train_s_" + str(args.split) + "_f_" + str(args.fold) + ".csv")
    valid_csv = join(relative_fold_path, "valid_s_" + str(args.split) + "_f_" + str(args.fold) + ".csv")
    test_csv = join(relative_fold_path, "test_s_" + str(args.split) + "_f_" + str(args.fold) + ".csv")

    # Check if the csv files exist. If not, raise an error.
    if not all([os.path.isfile(train_csv), os.path.isfile(valid_csv), os.path.isfile(test_csv)]):
        raise ValueError("Missing *.cvs files ({}[{}], {}[{}], {}[{}])".format(train_csv, os.path.isfile(train_csv),
                                                                               valid_csv, os.path.isfile(valid_csv),
                                                                               test_csv, os.path.isfile(test_csv)))

    rootpath = get_rootpath_2_dataset(args)

    train_samples = csv_loader(train_csv, rootpath)
    valid_samples = csv_loader(valid_csv, rootpath)
    test_samples = csv_loader(test_csv, rootpath)

    # Just for debug to go fast.
    if DEBUG_MODE and (args.dataset == "Caltech-UCSD-Birds-200-2011"):
        reproducibility.force_seed(int(os.environ["MYSEED"]))
        warnings.warn("YOU ARE IN DEBUG MODE!!!!")
        train_samples = random.sample(train_samples, 100)
        valid_samples = random.sample(valid_samples, 5)
        test_samples = test_samples[:20]
        reproducibility.force_seed(int(os.environ["MYSEED"]))

    if DEBUG_MODE and (args.dataset == "glas"):
        reproducibility.force_seed(int(os.environ["MYSEED"]))
        warnings.warn("YOU ARE IN DEBUG MODE!!!!")
        train_samples = random.sample(train_samples, 20)
        valid_samples = random.sample(valid_samples, 5)
        test_samples = test_samples[:20]
        reproducibility.force_seed(int(os.environ["MYSEED"]))

    # TODO:
    #  Augment validation set by performing multiple transformations over the original data, and storing it on disc. See
    #  if it is possible to not-store on disc the transformed data and augment in a deterministic way on the fly.

    announce_msg("creating datasets and dataloaders")

    reproducibility.force_seed(int(os.environ["MYSEED"]))
    trainset = PhotoDataset(
        train_samples, args.dataset, args.name_classes, transform_tensor,
        set_for_eval=False, transform_img=train_transform_img, resize=args.resize, crop_size=args.crop_size,
        padding_size=args.padding_size, padding_mode=args.padding_mode, up_scale_small_dim_to=args.up_scale_small_dim_to
    )

    # TODO: find a better way to protect the reproducibility of operations that changes any random generator's state.
    #  We will call it: reproducibility armor. Functions/classes/operations that use a random generator should be
    #  independent from each other.
    reproducibility.force_seed(int(os.environ["MYSEED"]))
    validset = PhotoDataset(
        valid_samples, args.dataset, args.name_classes, transform_tensor,
        set_for_eval=False, transform_img=None, resize=args.resize, crop_size=None, padding_size=None if not
        args.pad_eval else args.padding_size, padding_mode=None if not args.pad_eval else args.padding_mode,
        force_div_32=True, up_scale_small_dim_to=args.up_scale_small_dim_to
    )

    reproducibility.force_seed(int(os.environ["MYSEED"]))
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=True, worker_init_fn=_init_fn, collate_fn=default_collate)

    reproducibility.force_seed(int(os.environ["MYSEED"]))
    valid_loader = DataLoader(
        validset, batch_size=1, shuffle=False, num_workers=args.num_workers * FACTOR_MUL_WORKERS, pin_memory=True,
        collate_fn=default_collate, worker_init_fn=_init_fn
    )  # we need more workers since the batch size is 1, and set_for_eval is False (need more time to prepare a sample).
    reproducibility.force_seed(int(os.environ["MYSEED"]))

    # ################################ Instantiate models ########################################
    reproducibility.force_seed(int(os.environ["MYSEED"]))
    model = instantiate_models(args)

    # Check if we are using a user specific pre-trained model other than our pre-defined pre-trained models.
    # This can be used to to EVALUATE a trained model. You need to set args.max_epochs to -1 so no training is
    # performed. This is a hack to avoid creating other function to deal with LATER-evaluation after this code is done.
    # This script is intended for training. We evaluate at the end. However, if you missed something during the
    # training/evaluation (for example plot something over the predicted images), you do not need to re-train the
    # model. You can 1. specify the path to the pre-trained model. 2. Set max_epochs to -1. 3. Set strict to True. By
    # doing this, we load the pre-trained model, and, we skip the training loop, fast-forward to the evaluation.

    if hasattr(args, "path_pre_trained"):
        warnings.warn("You have asked to load a specific pre-trained model from {} .... [OK]".format(
            args.path_pre_trained))
        model = load_pre_pretrained_model(model=model, path_file=args.path_pre_trained, strict=args.strict)

    # best_model = deepcopy(model)  # Copy the model to device (0) before applying multi-gpu (in case it is one).
    # best_model.to(DEVICE)
    # best_state_dict = copy_model_state_dict_from_gpu_to_cpu(model)

    # Check if there are multiple GPUS.
    if ALLOW_MULTIGPUS:
        model = MyDataParallel(model)
        if args.batch_size < NBRGPUS:
            warnings.warn("You asked for MULTIGPU mode. However, your batch size {} is smaller than the number of "
                          "GPUs available {}. This is fine in practice. However, some GPUs will be idol. "
                          "This is just a warning .... [OK]".format(args.batch_size, NBRGPUS))
    model.to(DEVICE)
    # Copy the model's params.
    best_state_dict = deepcopy(model.state_dict())  # it has to be deepcopy.

    # ############################### Instantiate optimizer #################################
    reproducibility.force_seed(int(os.environ["MYSEED"]))
    optimizer, lr_scheduler = instantiate_optimizer(args, model)

    # ################################ Training ##################################
    reproducibility.force_seed(int(os.environ["MYSEED"]))
    tr_stats, tr_eval_stats, vl_stats = init_stats(train=True), init_stats(), init_stats()

    best_val_error = np.finfo(np.float32).max
    best_val_loss = np.finfo(np.float32).max
    best_epoch = 0

    # TODO: validate before start training.

    announce_msg("start training")
    reproducibility.force_seed(int(os.environ["MYSEED"]))
    tx0 = dt.datetime.now()

    for epoch in range(args.max_epochs):
        # TODO: IN THE FUTURE: DO NOT USE MAX_EPOCHS IN THE COMPUTATION OF THE CURRENT SEED!!!!
        # REPLACE IT WITH A CONSTANT (400 IN OUR CASE ON GLAS)
        reproducibility.force_seed(int(os.environ["MYSEED"]) + (epoch + 1) * 10000 + 400)
        trainset.set_up_new_seeds()
        reproducibility.force_seed(int(os.environ["MYSEED"]) + (epoch + 2) * 10000 + 400)
        validset.set_up_new_seeds()

        # Start the training with fresh seeds.
        reproducibility.force_seed(int(os.environ["MYSEED"]) + (epoch + 3) * 10000 + 400)

        if epoch >= args.epoch_start_erasing:  # activate the erasing.
            model.nbr_times_erase = model.nbr_times_erase_backup
        else:  # deactivate the erasing.
            model.nbr_times_erase = 0

        reproducibility.force_seed(int(os.environ["MYSEED"]) + (epoch + 4) * 10000 + 400)
        tr_stats = train_one_epoch(model, optimizer, train_loader, CRIT_TR, DEVICE, tr_stats, epoch,
                                   callback, training_log, ALLOW_MULTIGPUS=ALLOW_MULTIGPUS, NBRGPUS=NBRGPUS)

        if lr_scheduler:  # for > 1.1 : opt.step() then l_r_s.step().
            lr_scheduler.step(epoch)
        # TODO: Which criterion to use over the validation set? (in case this loss is used for model selection).
        # Eval validation set.
        reproducibility.force_seed(int(os.environ["MYSEED"]) + (epoch + 5) * 10000 + 400)
        vl_stats, stats_vl_now, pred_vl = validate(model=model, dataset=validset, dataloader=valid_loader,
                                                   criterion=CRIT_EV,
                                                   device=DEVICE, stats=vl_stats, epoch=epoch,
                                                   callback=callback, log_file=training_log, name_set="valid")

        reproducibility.force_seed(int(os.environ["MYSEED"]) + (epoch + 6) * 10000 + 400)
        # Eval train set. (entire image not patches).
        # tr_eval_stats, stats_tr_ev_now, pred_tr_eval = validate(model=model, dataset=trainset_eval,
        #                                                         dataloader=train_eval_loader,
        #                                                         criterion=CRIT_EV, device=DEVICE,,
        #                                                         stats=tr_eval_stats, epoch=epoch, callback=callback,
        #                                                         log_file=training_log,  name_set="train")

        error_vl = vl_stats["errors"][-1]

        # TODO: Revise the model selection.
        if error_vl <= best_val_error:  # and loss <= best_val_loss:
            best_val_error = error_vl
            best_val_loss = vl_stats["loss_pos"][-1]
            best_state_dict = deepcopy(model.state_dict())  # it has to be deepcopy.

            # Expensive operation: disc I/O.
            # torch.save(best_model.state_dict(), join(OUTD, "best_model.pt"))
            best_epoch = epoch

        if PLOT_STATS:
            print("Plotting stats ...")
            plot_curves(
                tr_stats, join(OUTD_TR.folder, "train.png"),
                "Train stats. Best epoch: {}.".format(best_epoch))
            plot_curves(vl_stats, join(OUTD_VL.folder, "validation.png"),
                        "Eval (validation set) stats. Best epoch: {}.".format(best_epoch))
            # plot_curves(
            #     tr_eval_stats, join(OUTD_TR.folder, "train-ev.png"),
            #     "Eval (train set) stats. . Best epoch: {}.".format(best_epoch))

            # # Plot the probability dist.
            # plot_hist_probs_pos_neg({"probs_pos": pred_tr_eval["probs_pos"],
            #                          "probs_neg": pred_tr_eval["probs_neg"]},
            #                         path=join(OUTD_TR.prob_evol, "train_prob+-_{}.png".format(epoch)),
            #                         epoch=epoch, title="Train eval. Probs.dist.")
            # plot_hist_probs_pos_neg({"probs_pos": pred_vl["probs_pos"],
            #                          "probs_neg": pred_vl["probs_neg"]},
            #                         path=join(OUTD_VL.prob_evol, "valid_prob+-_{}.png".format(epoch)),
            #                         epoch=epoch, title="Validation. Probs.dist.")
            #
            # # Plot ROC curves.
            # plot_roc_curve(y_mask=stats_tr_ev_now["for_roc"]["y_mask"],
            #                y_hat_mask=stats_tr_ev_now["for_roc"]["y_hat_mask"],
            #                epoch=epoch, path=join(OUTD_TR.roc_evol, "train_roc_{}.png".format(epoch)),
            #                title="Train. ROC.", plot_extreme=True)
            # plot_roc_curve(y_mask=stats_vl_now["for_roc"]["y_mask"],
            #                y_hat_mask=stats_vl_now["for_roc"]["y_hat_mask"],
            #                epoch=epoch, path=join(OUTD_VL.roc_evol, "train_roc_{}.png".format(epoch)),
            #                title="Validation. ROC.", plot_extreme=True)

    # =====================================================================================
    #                                   DO CLOSING-STUFF BEFORE LEAVING
    # =====================================================================================
    # Classification errors using the best model over: train/valid/test sets.
    # Train: needs to reload it with eval-transformations, not train-transformations.

    # Reset the models parameters to the best found ones.
    model.load_state_dict(best_state_dict)

    # We need to do each set sequentially to free the memory.

    announce_msg("End training. Time: {}".format(dt.datetime.now() - tx0))

    tx0 = dt.datetime.now()

    reproducibility.force_seed(int(os.environ["MYSEED"]))

    plot_curves(
        tr_stats, join(OUTD_TR.folder, "train.png"),
        "Train stats. Best epoch: {}.".format(best_epoch))
    plot_curves(vl_stats, join(OUTD_VL.folder, "validation.png"),
                "Eval (validation set) stats. Best epoch: {}.".format(best_epoch))

    announce_msg("start final processing stage")

    if DEBUG_MODE and (args.dataset == "Caltech-UCSD-Birds-200-2011"):
        reproducibility.force_seed(int(os.environ["MYSEED"]))
        testset = PhotoDataset(
            test_samples, args.dataset, args.name_classes, transform_tensor,
            set_for_eval=False, transform_img=None, resize=args.resize, crop_size=None, padding_size=None if not
            args.pad_eval else args.padding_size, padding_mode=None if not args.pad_eval else args.padding_mode,
            force_div_32=True, up_scale_small_dim_to=args.up_scale_small_dim_to, do_not_save_samples=True
        )

        reproducibility.force_seed(int(os.environ["MYSEED"]))
        test_loader = DataLoader(
            testset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True,
            collate_fn=default_collate,
            worker_init_fn=_init_fn
        )

        reproducibility.force_seed(int(os.environ["MYSEED"]))
        final_processing(
            model,
            test_loader,
            testset,
            "Test",
            validate, CRIT_EV, DEVICE, best_epoch, callback, results_log, OUTD, args, save_pred_for_later_comp=True
        )
        reproducibility.force_seed(int(os.environ["MYSEED"]))

        # Move the state dict of the best model into CPU, then save it.
        best_state_dict_cpu = copy_model_state_dict_from_gpu_to_cpu(model)
        torch.save(best_state_dict_cpu, join(OUTD, "best_model.pt"))

        announce_msg("End final processing. Time: {}".format(dt.datetime.now() - tx0))

        announce_msg("*END*")

        sys.exit()

    del trainset
    del train_loader

    reproducibility.force_seed(int(os.environ["MYSEED"]))
    final_processing(
        model,
        valid_loader,
        validset,
        "Validation",
        validate, CRIT_EV, DEVICE, best_epoch, callback, results_log, OUTD, args, save_pred_for_later_comp=False
    )
    del validset
    del valid_loader

    reproducibility.force_seed(int(os.environ["MYSEED"]))
    testset = PhotoDataset(
        test_samples, args.dataset, args.name_classes, transform_tensor,
        set_for_eval=False, transform_img=None, resize=args.resize, crop_size=None, padding_size=None if not
        args.pad_eval else args.padding_size, padding_mode=None if not args.pad_eval else args.padding_mode,
        force_div_32=True, up_scale_small_dim_to=args.up_scale_small_dim_to, do_not_save_samples=True
    )

    reproducibility.force_seed(int(os.environ["MYSEED"]))
    test_loader = DataLoader(
        testset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True,
        collate_fn=default_collate,
        worker_init_fn=_init_fn
    )

    reproducibility.force_seed(int(os.environ["MYSEED"]))
    final_processing(
        model,
        test_loader,
        testset,
        "Test",
        validate, CRIT_EV, DEVICE, best_epoch, callback, results_log, OUTD, args, save_pred_for_later_comp=True
    )

    del testset
    del test_loader

    reproducibility.force_seed(int(os.environ["MYSEED"]))
    trainset_eval = PhotoDataset(
        train_samples, args.dataset, args.name_classes, transform_tensor,
        set_for_eval=False, transform_img=None, resize=args.resize, crop_size=None, padding_size=None if not
        args.pad_eval else args.padding_size, padding_mode=None if not args.pad_eval else args.padding_mode,
        force_div_32=True, up_scale_small_dim_to=args.up_scale_small_dim_to, do_not_save_samples=True
    )

    reproducibility.force_seed(int(os.environ["MYSEED"]))
    train_eval_loader = DataLoader(
        trainset_eval, batch_size=1, shuffle=False, num_workers=0, pin_memory=True,
        collate_fn=default_collate, worker_init_fn=_init_fn
    )

    reproducibility.force_seed(int(os.environ["MYSEED"]))
    final_processing(
        model,
        train_eval_loader,
        trainset_eval,
        "Train",
        validate, CRIT_EV, DEVICE, best_epoch, callback, results_log, OUTD, args, save_pred_for_later_comp=False
    )

    # Save train statistics (train, valid)
    stats_to_dump = {
        "train": tr_stats,
        "valid": vl_stats,
        "eval_train": tr_eval_stats
    }
    with open(join(OUTD, "train_stats.pkl"), "wb") as fout:
        pkl.dump(stats_to_dump, fout, protocol=pkl.HIGHEST_PROTOCOL)

    # Move the state dict of the best model into CPU, then save it.
    best_state_dict_cpu = copy_model_state_dict_from_gpu_to_cpu(model)
    torch.save(model.state_dict(), join(OUTD, "best_model.pt"))
    announce_msg("End final processing. Time: {}".format(dt.datetime.now() - tx0))

    announce_msg("*END*")
    # ======================================================================================
    #                                                END
    # ======================================================================================

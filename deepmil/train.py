import os
import random

import torch

import datetime as dt
import tqdm
import numpy as np
from scipy.special import softmax
import torch
import torch.nn.functional as F

from tools import AverageMeter
from tools import log

import reproducibility


def train_one_epoch(model, optimizer, dataloader, criterion, device, tr_stats, epoch=0, callback=None,
                    log_file=None):
    """
    Perform one epoch of training.
    :param model:
    :param optimizer:
    :param dataloader:
    :param criterion:
    :param device:
    :param epoch:
    :param callback:
    :param log_file:
    :return:
    """
    model.train()

    total_loss, loss_pos, loss_neg = AverageMeter(), AverageMeter(), AverageMeter()
    loss_class_seg, errors = AverageMeter(), AverageMeter()

    length = len(dataloader)
    t0 = dt.datetime.now()

    for i, (data, masks, labels) in tqdm.tqdm(enumerate(dataloader), ncols=80, total=length):
        reproducibility.force_seed(int(os.environ["MYSEED"]) + epoch)

        data = data.to(device)
        labels = labels.to(device)

        model.zero_grad()

        t_l, l_p, l_n, l_c_s = 0., 0., 0., 0.

        # Optimization:
        if model.nbr_times_erase == 0:  # no erasing.
            output = model(data)  # --> out_pos, out_neg, masks
            _, _, _, scores_seg, _ = output
            l_c_s = criterion.loss_class_head_seg(scores_seg, labels)

            loss = criterion(output, labels)
            t_l, l_p, l_n = loss
            # print("\t \t \t \t \t {} \t {} \t {} \t {}".format(t_l.item(), l_p.item(), l_n.item(), l_a.item()))
            t_l = t_l + l_c_s
            t_l.backward()
        else:  # we need to erase some times.
            # Compute the cumulative mask.
            l_c_s = 0.
            l_c_s_per_sample = None
            m_pos = None
            data_safe = torch.zeros_like(data)
            data_safe = data_safe.copy_(data)
            history = torch.ones_like(labels).type(torch.float)  # init. history tracker coefs. to 1.
            # if the model predicts the wrong label, we set forever the trust for this sample to 0.
            # for er in range(model.nbr_times_erase + 1):
            er = 0
            while history.sum() > 0:

                if er >= model.nbr_times_erase:  # if we exceed the maximum, stop looping. We are not looping
                    # forever!! aren't we?
                    break
                mask, scores_seg, _ = model.segment(data)  # mask is detached! (mask is continuous)
                mask, _, _ = model.get_mask_xpos_xneg(data, mask)  # mask = M+.
                probs_seg = criterion.softmax(scores_seg)

                l_c_s_tmp = criterion.loss_class_head_seg_red_none(scores_seg, labels)
                l_c_s_tmp.mean().backward()

                # avoid maintaining the previous graph. Therefore,
                # cut the dependency.
                l_c_s_tmp = l_c_s_tmp.detach()
                probs_seg = probs_seg.detach()

                l_c_s += l_c_s_tmp.mean()  # for tracking only.

                # Update the mask (m_pos: M+): The negative mask is expected to contain all the non-discriminative
                # regions. However, it may still contain some discriminative parts. In order to find them, we apply M-
                # over the input image (erase the found discriminative parts) and try to localize NEW discriminative
                # regions.

                if er == 0:  # if the first time, create the tracking mask.
                    m_pos = torch.zeros_like(mask)
                    l_c_s_per_sample = torch.zeros_like(l_c_s_tmp)
                    l_c_s_per_sample = l_c_s_per_sample.copy_(l_c_s_tmp)

                trust = torch.ones_like(labels).type(torch.float)  # init. trust coefs. to 1.
                p_y, pred = torch.max(probs_seg, dim=1)

                # overall trust:
                decay = np.exp(-float(er) / model.sigma_erase)
                # per-sample trust:
                check_loss = (l_c_s_tmp <= l_c_s_per_sample).type(torch.float)
                check_label = (pred == labels).type(torch.float)

                trust *= decay * check_label * check_loss * p_y

                # Update the history
                history = torch.min(check_label, history)

                # Apply the history to the trust:
                trust *= history

                trust = trust.view(trust.size()[0], 1, 1, 1)
                m_pos_tmp = trust * mask
                m_pos = torch.max(m_pos, m_pos_tmp)  # accumulate the masks.
                # Apply the cumulative negative mask over the image
                data = data * (1 - m_pos) * (trust != 0).type(torch.float) + data * (trust == 0).type(torch.float)
                er += 1

            l_c_s /= (model.nbr_times_erase + 1)

            # Now: m_neg contains the smallest negative area == largest positive area.
            # compute x_pos, x_neg
            m_neg = 1 - m_pos
            x_neg = data_safe * m_neg
            x_pos = data_safe * m_pos

            # Classify
            out_pos = model.classify(x_pos)
            out_neg = model.classify(x_neg)

            output = out_pos, out_neg, None, None, None
            loss = criterion(output, labels)
            t_l, l_p, l_n = loss
            t_l.backward()
            t_l += l_c_s

        # Update params.
        optimizer.step()
        # End optimization

        total_loss.append(t_l.item())
        loss_pos.append(l_p.item())
        loss_neg.append(l_n.item())
        loss_class_seg.append(l_c_s.item())

        errors.append((output[0][0].argmax(dim=1) != labels).float().mean().item() * 100.)  # error over the minibatch.

        if callback and ((i + 1) % callback.fre == 0 or (i + 1) == length):
            callback.scalar("Train_loss", i / length + epoch, total_loss.last_avg)
            callback.scalar("Train_error", i / length + epoch, errors.lat_avg)

    to_write = "Train epoch {:>2d}: Total L.avg: {:.5f}, Pos.L.avg: {:.5f}, Neg.L.avg: {:.5f}, " \
               "Cl.Seg.L.avg: {:.5f}, LR {}, t:{}".format(
                epoch, total_loss.avg, loss_pos.avg, loss_neg.avg, loss_class_seg.avg,
                ['{:.2e}'.format(group["lr"]) for group in optimizer.param_groups], dt.datetime.now() - t0
                )
    print(to_write)
    if log_file:
        log(log_file, to_write)

    # Update stats:
    tr_stats["total_loss"] = np.append(tr_stats["total_loss"], np.array(total_loss.values))
    tr_stats["loss_pos"] = np.append(tr_stats["loss_pos"], np.array(loss_pos.values))
    tr_stats["loss_neg"] = np.append(tr_stats["loss_neg"], np.array(loss_neg.values))
    tr_stats["loss_class_seg"] = np.append(tr_stats["loss_class_seg"], np.array(loss_class_seg.values))
    tr_stats["errors"] = np.append(tr_stats["errors"], np.array(errors.values))
    return tr_stats


def validate(model, dataset, dataloader, criterion, device, stats, epoch=0, callback=None, log_file=None,
             name_set=""):
    """
    Perform a validation over the validation set. Assumes a batch size of 1. (images do not have the same size,
    so we can't stack them in one tensor).
    Validation samples may be large to fit all in the GPU at once.

    Note: criterion is deppmil.criteria.TotalLossEval().
    """
    # TODO: [FUTURE] find a way to do final processing within the loop below such as drawing and all that. This will
    #  avoid STORING the results of validating over huge datasets, THEN, process samples one by one!!! it is not
    #  efficient. This should be an option to be activated/deactivated when needed. For instance, during validation,
    #  it is not necessary, while at the end, it is necessary.
    model.eval()

    total_loss, loss_pos, loss_neg = AverageMeter(), AverageMeter(), AverageMeter()
    loss_class_seg, loss_dice = AverageMeter(), AverageMeter()
    errors = AverageMeter()

    length = len(dataloader)
    predictions = np.zeros(length, dtype=int)
    labels = np.zeros(length)
    probs = np.zeros(length)  # prob. of the predicted class (using positive region).
    probs_pos = np.zeros((length, model.num_classes))  # prob. over the positive region.
    probs_neg = np.zeros((length, model.num_classes))  # prob. over the negative region.
    masks_pred = []
    t0 = dt.datetime.now()

    with torch.no_grad():
        for i, (data, mask, label) in tqdm.tqdm(enumerate(dataloader), ncols=80, total=length):
            reproducibility.force_seed(int(os.environ["MYSEED"]) + epoch)

            assert data.size()[0] == 1, "Expected a batch size of 1. Found `{}`  .... [NOT OK]".format(data.size()[0])

            labels[i] = label.item()  # batch size 1.
            data = data.to(device)
            label = label.to(device)

            mask_t = [m.to(device) for m in mask]

            output = model(data)  # --> out_pos, out_neg, masks

            out_pos = output[0][0][0]  # scores: take the first element of the batch.
            out_neg = output[1][0][0]  # scores: take the first element of the batch.
            assert out_pos.ndimension() == 1, "We expected only 1 dim. We found {}. Make sure you are using abatch " \
                                              "size of 1. .... [NOT OK]".format(out_pos.ndimension())

            pred_label = out_pos.argmax()
            predictions[i] = int(pred_label.item())

            scores_pos = softmax(out_pos.cpu().detach().numpy())
            scores_neg = softmax(out_neg.cpu().detach().numpy())
            probs_pos[i] = scores_pos
            probs_neg[i] = scores_neg
            probs[i] = scores_pos[predictions[i]]
            mask_pred = torch.squeeze(output[2]).cpu().detach().numpy()  # predicted mask.
            # check sizes of the mask:
            _, h, w = mask_t[0].size()
            hp, wp = mask_pred.shape

            if dataset.dataset_name == "Caltech-UCSD-Birds-200-2011":
                # Remove the padding if is there was any. (the only one allowed: force_div_32)
                assert dataset.padding_size is None, "dataset.padding_size is supposed to be None. We do not support" \
                                                     "padding of this type for this dataset."

                # Important: we assume that dataloader of this dataset is not shuffled to make this access using (i)
                # valid. If shuffled, the access is not correct.
                w_mask_no_pad_forced, h_mask_no_pad_forced = dataset.original_images_size[i]

                if dataset.force_div_32:
                    # Find the size of the mask without padding.
                    w_up, h_up = dataset.get_upscaled_dims(w_mask_no_pad_forced, h_mask_no_pad_forced,
                                                           dataset.up_scale_small_dim_to)
                    # Remove the padded parts by cropping at the center.
                    mask_pred = mask_pred[int(hp / 2) - int(h_up / 2): int(hp / 2) + int(h_up / 2) + (h_up % 2),
                                int(wp / 2) - int(w_up / 2):int(wp / 2) + int(w_up / 2) + (w_up % 2)]
                    assert mask_pred.shape[0] == h_up, "h_up={}, mask_pred.shape[0]={}. Expected to be the same." \
                                                       "[Not OK]".format(h_up, mask_pred.shape[0])
                    assert mask_pred.shape[1] == w_up, "w_up={}, mask_pred.shape[1]={}. Expected to be the same." \
                                                       "[Not OK]".format(w_up, mask_pred.shape[1])
                # Now, downscale the predicted mask to the size of the true mask. We use
                # torch.nn.functional.interpolate.
                msk_pred_torch = torch.from_numpy(mask_pred).view(1, 1, mask_pred.shape[0], mask_pred.shape[1])
                mask_pred = F.interpolate(msk_pred_torch, size=(h_mask_no_pad_forced, w_mask_no_pad_forced),
                                          mode="bilinear", align_corners=True).squeeze().numpy()

                # Now get the correct sizes:
                hp, wp = mask_pred.shape
                assert hp == h, "hp={}, h={} are suppored to be the same! .... [NOT OK]".format(hp, h)
                assert wp == w, "wp={}, w={} are suppored to be the same! .... [NOT OK]".format(wp, w)

            if (h != hp) or (w != wp):  # This means that we have padded the input image.
                # We crop the predicted mask in the center.
                mask_pred = mask_pred[int(hp / 2) - int(h / 2): int(hp / 2) + int(h / 2) + (h % 2),
                                      int(wp / 2) - int(w / 2):int(wp / 2) + int(w / 2) + (w % 2)]
            masks_pred.append(mask_pred)

            loss = criterion(output, label, mask_t)
            t_l, l_p, l_n, l_d, l_c_s = loss

            total_loss.append(t_l.item())
            loss_pos.append(l_p.item())
            loss_neg.append(l_n.item())
            loss_class_seg.append(l_c_s.item())
            loss_dice.append(l_d.item())

            errors.append((pred_label != label).item())

    if callback:
        callback.scalar('Val_loss', epoch + 1, total_loss.avg)
        callback.scalar('Val_error', epoch + 1, errors.avg)

    to_write = ">>>>>>>>>>>>>>>>>> Total L.avg: {:.5f}, Pos.L.avg: {:.5f}, Neg.L.avg: {:.5f}, " \
               "Cl.Seg.L.avg: {:.5f}, D.L.avg: {:.5f}, Error.avg: {:.2f}, t:{}, Eval {} epoch {:>2d}, " \
               "".format(
                total_loss.avg, loss_pos.avg, loss_neg.avg, loss_class_seg.avg,
                loss_dice.avg, errors.avg * 100, dt.datetime.now() - t0, name_set, epoch
    )
    print(to_write)
    if log_file:
        log(log_file, to_write)

    # Update stats
    stats["total_loss"] = np.append(stats["total_loss"], np.array(total_loss.values))
    stats["loss_pos"] = np.append(stats["loss_pos"], np.array(loss_pos.values))
    stats["loss_neg"] = np.append(stats["loss_neg"], np.array(loss_neg.values))
    stats["loss_class_seg"] = np.append(stats["loss_class_seg"], np.array(loss_class_seg.values))
    stats["errors"] = np.append(stats["errors"], np.array(errors.values).mean() * 100)
    stats["loss_dice"] = np.append(stats["loss_dice"], np.array(loss_dice.values).mean() * 100)
    pred = {
        "predictions": predictions,
        "labels": labels,
        "probs": probs,
        "masks": masks_pred,
        "probs_pos": probs_pos,
        "probs_neg": probs_neg
    }

    # Collect stats from only this epoch. (can be useful to plot distributions since we lose the actual stats due to
    # the above update!!!!)
    stats_now = {
        "total_loss": np.array(total_loss.values),
        "loss_pos": np.array(loss_pos.values),
        "loss_neg": np.array(loss_neg.values),
        "loss_class_seg": np.array(loss_class_seg.values),
        "errors": np.array(errors.values).mean() * 100,
        "loss_dice": np.array(loss_dice.values)
    }

    return stats, stats_now, pred

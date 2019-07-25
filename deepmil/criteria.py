import torch
import torch.nn as nn

__all__ = ["TotalLoss", "TotalLossEval"]


class TotalLoss(nn.Module):
    """
    Loss.
    """
    def __init__(self):
        """
        Init. function.
        """
        super(TotalLoss, self).__init__()

        # self.const = torch.tensor([lam], requires_grad=False).float()
        # self.register_buffer("lam", self.const)

        self.CE = nn.CrossEntropyLoss(reduction="mean")  # The cross entropy loss.
        self.CE_red_none = nn.CrossEntropyLoss(reduction="none")  # The cross entropy loss (loss per sample).
        self.LSM = nn.LogSoftmax(dim=1)  # The log-softmax.
        self.softmax = nn.Softmax(dim=1)  # Compute the softmax.

    def loss_class_head_seg(self, scores_seg, labels):
        """
        Compute the classification loss over the segmentation head.
        :param scores_seg: matrix of size (n, c) of the scores of each class for each sample in the batch size. c is
        the number of classes.
        :param labels: vector of true image labels (dim: n).
        :return:
        """
        return self.CE(scores_seg, labels)

    def loss_class_head_seg_red_none(self, scores_seg, labels):
        """
        Compute the per-sample classification loss over the segmentation head.
        :param scores_seg: matrix of size (n, c) of the scores of each class for each sample in the batch size. c is
        the number of classes.
        :param labels: vector of true image labels (dim: n).
        :return:
        """
        return self.CE_red_none(scores_seg, labels)

    def shared_losses(self, netoutput, labels):
        """
        Compute the shared losses between train/eval: loss over the positive region, loss over the negative region.
        :param netoutput:
        :param labels:
        :return:
        """
        out_pos, out_neg, _, _, _ = netoutput
        scores_pos, maps_pos = out_pos
        scores_neg, maps_neg = out_neg

        # Loss of positive regions at the Classification head.
        loss_pos = self.CE(scores_pos, labels)

        # Loss of negative regions at the classification head.
        log_softmx_scores_neg = self.LSM(scores_neg)
        loss_neg = - (log_softmx_scores_neg.mean(dim=1)).mean()

        return loss_pos, loss_neg

    def forward(self, netoutput, labels):
        """
        Performs forward function: computes the losses.
        Input:
            netoutput: tuple, the network output.
                (out_pos, out_neg, mask).
            labels: tensor of long int. It contains the true image labels.
        """
        out_pos, out_neg, _, _, _ = netoutput
        loss_pos, loss_neg = self.shared_losses(netoutput, labels)

        # Total loss
        total_loss = loss_pos + loss_neg

        return total_loss.squeeze(), loss_pos.squeeze(), loss_neg.squeeze()

    def __str__(self):
        return "{}()".format(self.__class__.__name__,)


class TotalLossEval(TotalLoss):
    """
    Total loss for evaluation. Same as `TotalLoss`; we add computing Dice index.
    """
    def __init__(self):
        """
        Init. function.
        """
        super(TotalLossEval, self).__init__()

    def dice(self, pred, target):
        """
        Compute Dice index.

        Notes:
        1. Since this function is called only for evaluation, the list of tensors contains only one tensor
        (batch size = 1).
        2. The predicted masks must have the same size as the target. Otherwise, the predicted mask will be cropped
        at the center (the only reason of the difference is that the input image was padded which leads to a padded
        predicted mask with the same size as the image. Therefore, crop).


        :param pred: list of pytroch tensor of size (1, h, w). The predicted masks.
        :param target: list of pytroch tensors of size (1, h, w). The true masks.
        :return: dice index, and the predicted mask (continuous not binary) with the same size of the target mask.
        """
        msg = "`pred` should be a list of tensors. You provided `{}` .... [NOT OK]".format(type(pred))
        assert isinstance(pred, list), msg

        msg = "`target` should be a list of tensors. You provided `{}` .... [NOT OK]".format(type(target))
        assert isinstance(target, list), msg

        assert len(pred) == 1, "`pred` should contain only one tensor. We found `{}` .... [NOT OK]".format(len(pred))

        msg = "`target` should contain only one tensor. We found `{}` .... [NOT OK]".format(len(target))
        assert len(target) == 1, msg

        mask = torch.squeeze(pred[0])
        mask_t = torch.squeeze(target[0])
        b, d, _, _ = pred[0].size()

        if mask.size() != mask_t.size():  # This means that we have padded the input image.
            # We crop the predicted mask in the center.
            h, w = mask_t.size()
            hp, wp = mask.size()
            mask = mask[int(hp / 2) - int(h / 2): int(hp / 2) + int(h / 2) + (h % 2),
                        int(wp / 2) - int(w / 2):int(wp / 2) + int(w / 2) + (w % 2)]

        msg = "Masks need to have the same size. `pred`: {}, `target`: {}  .... [NOT OK]".format(
            mask.size(), mask_t.size())
        assert mask.size() == mask_t.size(), msg

        # Make the predicted mask BINARY:
        mask_bin = ((mask >= 0.5) * 1.).float()  # The derivative of this is ZERO. So do not use it for training.

        # Compute Dice index.
        pflat = mask_bin.view(-1)
        tflat = mask_t.view(-1)
        intersection = (pflat * tflat).sum()

        h, w = mask.size()

        return (2. * intersection) / (pflat.sum() + tflat.sum()), mask.view(b, d, h, w)

    def forward(self, netoutput, labels, masks_t):
        """
        Assumes batch_size = 1. (since the images do not have the same size)

        :param netoutput: out_pos, out_neg, masks.
        :param labels: tensor of the image labels.
        :param masks_t: list of tensors containing the true masks.
        :return: total_loss, loss_pos, loss_neg, f1pos, f1neg, loss_class_seg.
        """
        # Compute Dice index, and get the adjusted predicted mask (cropped if necessary. We do not use the padded
        # mask from the net output.).
        out_pos, out_neg, masks, scores_seg, _ = netoutput
        # We do not compute dice here since we need to do some  upsacling operations over it depending on the dataset.
        #TODO: Remove entirely these lines, and make dice computation universel in a sens the two input have the same
        # dimension and there is no need to do any cropping or other operations.

        # Compute F1 (dice index) over positive regions.
        # f1pos, masks = self.dice([masks], masks_t)
        # Compute F1 over the negative regions.
        # f1neg, mask = self.dice([1. - masks], [1. - m for m in masks_t])

        # Compute loss pos, neg areas.
        loss_pos, loss_neg = self.shared_losses(netoutput, labels)

        loss_class_seg = self.loss_class_head_seg(scores_seg, labels)
        # Total loss
        total_loss = loss_class_seg + loss_pos + loss_neg

        return total_loss, loss_pos, loss_neg, loss_class_seg


# ====================== TEST =========================================

def test_TotalLoss():
    loss = TotalLoss()
    print("Testing {}".format(loss))
    cuda = 0
    DEVICE = torch.device("cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(int(cuda))
    loss.to(DEVICE)
    num_classes = 2
    b, h, w = 16, 200, 200
    masks = torch.rand(b, 1, h, w).to(DEVICE)
    out_pos = (torch.rand(b, num_classes).to(DEVICE), torch.rand(b, num_classes, h, w).to(DEVICE))
    out_neg = (torch.rand(b, num_classes).to(DEVICE), torch.rand(b, num_classes, h, w).to(DEVICE))
    scores_seg = (torch.rand(b, num_classes)).to(DEVICE)
    maps_seg = (torch.rand(b, num_classes, 10, 10)).to(DEVICE)
    netoutput = (out_pos, out_neg, masks, scores_seg, maps_seg)
    labels = torch.empty(b, dtype=torch.long).random_(2).to(DEVICE)

    print("Loss class at head: {}".format(loss.loss_class_head_seg(scores_seg, labels)))
    losses = loss(netoutput, labels)
    for l in losses:
        print(l, l.size())


def test_TotalLossEval():
    loss = TotalLossEval()
    print("Testing {}".format(loss))
    cuda = 1
    DEVICE = torch.device("cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(int(cuda))
    loss.to(DEVICE)
    num_classes = 2
    b, h, w = 1, 200, 200
    masks = torch.rand(b, 1, h, w).to(DEVICE)
    out_pos = (torch.rand(b, num_classes).to(DEVICE), torch.rand(b, num_classes, h, w).to(DEVICE))
    out_neg = (torch.rand(b, num_classes).to(DEVICE), torch.rand(b, num_classes, h, w).to(DEVICE))
    scores_seg = (torch.rand(b, num_classes)).to(DEVICE)
    maps_seg = (torch.rand(b, num_classes, 10, 10)).to(DEVICE)
    netoutput = (out_pos, out_neg, masks, scores_seg, maps_seg)
    labels = torch.empty(b, dtype=torch.long).random_(2).to(DEVICE)

    losses = loss(netoutput, labels, [masks])
    for l in losses:
        print(l, l.size())


if __name__ == "__main__":
    test_TotalLoss()
    test_TotalLossEval()


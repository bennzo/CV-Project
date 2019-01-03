import torch
import torch.nn as nn
from torch.nn.functional import smooth_l1_loss

class RetinaLoss(nn.Module):
    def __init__(self, alpha_balanced=False):
        super(RetinaLoss, self).__init__()
        self.gamma = 2
        self.alpha = 0.25
        self.ab = alpha_balanced


    def forward(self, pred_annots, gt_annots):
        batch_size = gt_annots.shape[0]
        pred_offsets, pred_classes = pred_annots
        gt_offsets = gt_annots[:, :, :4]
        gt_classes = gt_annots[:, :, 4:]

        loc_loss = []
        cls_loss = []

        for i in range(batch_size):
            pred_offset, pred_class = pred_offsets[i], pred_classes[i]
            gt_offset, gt_class = gt_offsets[i], gt_classes[i]
            inc_idxs = gt_class[:,0] > -1
            pos_idxs = gt_class.sum(dim=1) > 0

            pos_hits = pos_idxs.sum().float().cuda()

            # Class Loss
            class_targets = gt_class[inc_idxs]
            neg_class_targets = 1 - class_targets
            class_probs = pred_class[inc_idxs].clamp(1e-4, 1.0 - 1e-4)      # TEST
            # class_probs = pred_class[inc_idxs]
            neg_class_probs = 1 - class_probs

            alpha_weight = self.alpha*class_targets + (1-self.alpha)*neg_class_targets
            focal_weight = torch.pow(neg_class_probs*class_targets + class_probs*neg_class_targets, self.gamma)

            focal_loss = -1 * (class_targets*torch.log(class_probs) + neg_class_targets*torch.log(neg_class_probs)) * focal_weight * alpha_weight
            class_loss = focal_loss.sum() / pos_hits.clamp(min=1)

            cls_loss.append(class_loss)

            # Localization Loss
            # TODO: Normalize ground truth regression targets as in FastRCNN
            if pos_hits > 0:
                pred_offsets_pos = pred_offset[pos_idxs]
                gt_offsets_pos = gt_offset[pos_idxs]
                localization_loss = smooth_l1_loss(pred_offsets_pos, gt_offsets_pos)
            else:
                localization_loss = torch.tensor(0).float().cuda()

            loc_loss.append(localization_loss)

        return torch.stack(loc_loss).mean(), torch.stack(cls_loss).mean()


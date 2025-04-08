import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=2, beta=4):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, gt):
        pred = pred.sigmoid().clamp(min=1e-4, max=1-1e-4)

        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        neg_weights = torch.pow(1 - gt, self.beta)

        pos_loss = -torch.log(pred) * torch.pow(1 - pred, self.alpha) * pos_inds
        neg_loss = -torch.log(1 - pred) * torch.pow(pred, self.alpha) * neg_weights * neg_inds

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        loss = (pos_loss + neg_loss) / (num_pos + 1e-6)
        return loss


class MaskedL1Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, mask):
        """
        pred: [B, 2, H, W]
        target: [B, 2, H, W]
        mask: [B, 1, H, W] — бинарная маска (1.0 в центрах объектов)
        """
        mask = mask.expand_as(pred)  # [B, 2, H, W]
        loss = F.l1_loss(pred, target, reduction='none')  # [B, 2, H, W]
        loss = loss * mask
        return loss.sum() / (mask.sum() + 1e-6)


class TotalLoss(nn.Module):
    def __init__(self,
                 weight_cls=1.0,
                 weight_center=1.0,
                 weight_size=0.1,
                 weight_offset=1.0):
        super().__init__()
        self.cls_loss_fn = FocalLoss()
        self.center_loss_fn = FocalLoss()
        self.size_loss_fn = MaskedL1Loss()
        self.offset_loss_fn = MaskedL1Loss()

        self.weight_cls = weight_cls
        self.weight_center = weight_center
        self.weight_size = weight_size
        self.weight_offset = weight_offset

    def forward(self, preds, targets):
        heatmap = targets['heatmap']
        mask = heatmap.amax(dim=1, keepdim=True).eq(1).float()

        cls_loss = sum([self.cls_loss_fn(p, heatmap) for p in preds['cls']]) / len(preds['cls'])
        center_loss = sum([self.center_loss_fn(p, heatmap) for p in preds['center']]) / len(preds['center'])
        size_loss = sum([self.size_loss_fn(p, targets['size'], mask) for p in preds['size']]) / len(preds['size'])
        offset_loss = sum([self.offset_loss_fn(p, targets['offset'], mask) for p in preds['offset']]) / len(preds['offset'])

        total = (
            self.weight_cls * cls_loss +
            self.weight_center * center_loss +
            self.weight_size * size_loss +
            self.weight_offset * offset_loss
        )

        return {
            "total": total,
            "cls": cls_loss.item(),
            "center": center_loss.item(),
            "size": size_loss.item(),
            "offset": offset_loss.item()
        }
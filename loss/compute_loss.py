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

        num_pos = max(pos_inds.float().sum(), 1.0) #TODO
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
                 weight_size=1.0,
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

    def forward(self, preds, targets, debug=False):
        
        heatmaps = targets['heatmap']     # List[L] of [B, C, H, W]
        center = targets['center']    # List[L] of [B, 1, H, W]
        sizes = targets['size']        # List[L] of [B, 2, H, W]
        offsets = targets['offset']      # List[L] of [B, 2, H, W]
        masks = targets['mask']        # List[L] of [B, 1, H, W]  <-- ВАЖНО: из таргетов!

        if debug:
            for i in range(len(preds['cls'])):
                assert preds['cls'][i].shape == heatmaps[i].shape, f"cls shape mismatch at level {i}"
                assert preds['center'][i].shape == center[i].shape, f"center shape mismatch at level {i}"
                assert preds['size'][i].shape[2:] == sizes[i].shape[2:], f"size shape mismatch at level {i}"
                assert preds['offset'][i].shape[2:] == offsets[i].shape[2:], f"offset shape mismatch at level {i}"
            for idx, m in enumerate(masks):
                mask_sum = m.sum().item()
                msg = "[WARN]" if mask_sum == 0 else "[INFO]"
                print(f"{msg} Level {idx}: mask sum = {mask_sum}")

        
        # Подсчёт потерь по каждому уровню
        cls_loss = sum(self.cls_loss_fn(p, t) for p, t in zip(preds['cls'], heatmaps)) / len(preds['cls'])
        center_loss = sum(self.center_loss_fn(p, t) for p, t in zip(preds['center'], center)) / len(preds['center'])
        
        size_loss = 0.0
        offset_loss = 0.0
        for p_s, t_s, p_o, t_o, m in zip(preds['size'], sizes, preds['offset'], offsets, masks):
            size_loss   += self.size_loss_fn(p_s, t_s, m)
            offset_loss += self.offset_loss_fn(p_o, t_o, m)
        size_loss   /= len(preds['size'])
        offset_loss /= len(preds['offset'])

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
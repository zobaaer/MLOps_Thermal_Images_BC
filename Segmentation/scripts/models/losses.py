import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        num = 2 * (probs * targets).sum(dim=(1, 2, 3))
        den = (probs + targets).sum(dim=(1, 2, 3)) + self.smooth
        dice = num / den
        return 1 - dice.mean()


class ComboLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits, targets):
        return self.alpha * self.bce(logits, targets) + (1 - self.alpha) * self.dice(
            logits, targets
        )


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        """
        alpha: bilancia la classe positiva (1) vs negativa (0)
        gamma: quanto penalizzare gli esempi facili
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.clamp(min=1e-6, max=1 - 1e-6)  # per evitare log(0)
        targets = targets.float()

        # focal loss formula
        loss = -self.alpha * (
            targets * torch.pow(1 - probs, self.gamma) * torch.log(probs)
        ) - (1 - self.alpha) * (
            (1 - targets) * torch.pow(probs, self.gamma) * torch.log(1 - probs)
        )

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, smooth=1e-6):
        """
        alpha: penalizza i falsi positivi (FP)
        beta: penalizza i falsi negativi (FN)
        """
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.view(logits.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        TP = (probs * targets).sum(dim=1)
        FP = (probs * (1 - targets)).sum(dim=1)
        FN = ((1 - probs) * targets).sum(dim=1)

        tversky = (TP + self.smooth) / (
            TP + self.alpha * FP + self.beta * FN + self.smooth
        )
        return 1 - tversky.mean()


class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75, smooth=1e-6):
        """
        Focal Tversky Loss:
        - alpha: penalizza i falsi positivi
        - beta: penalizza i falsi negativi
        - gamma: focalizza su esempi difficili (come nella Focal Loss)
        """
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = torch.clamp(probs, 1e-4, 1.0 - 1e-4)  # per evitare log(0)
        probs = probs.view(logits.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        TP = (probs * targets).sum(dim=1)
        FP = (probs * (1 - targets)).sum(dim=1)
        FN = ((1 - probs) * targets).sum(dim=1)

        tversky_index = (TP + self.smooth) / (
            TP + self.alpha * FP + self.beta * FN + self.smooth
        )
        tversky_index = torch.clamp(tversky_index, 1e-4, 1.0 - 1e-4)
        focal_tversky = torch.pow((1 - tversky_index), self.gamma)
        return focal_tversky.mean()

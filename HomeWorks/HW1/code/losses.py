import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross-entropy with label smoothing."""
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        assert 0.0 <= smoothing < 1.0
        self.smoothing = smoothing

    def forward(self, logits, target):
        n_class = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (n_class - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), 1.0 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))


class CircleLoss(nn.Module):
    """Simple Circle Loss implementation for embedding + classification settings.
    Usage: provide normalized embeddings and label indices.
    Reference: "Circle Loss: A Unified Perspective of Pair Similarity Optimization" (Sun et al.)
    This implementation follows the pair-based formulation used for deep metric learning.
    """
    def __init__(self, m=0.25, gamma=256):
        super().__init__()
        self.m = m
        self.gamma = gamma

    def forward(self, embeddings, labels):
        # embeddings: (N, D) - expected normalized
        # labels: (N,)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        sim = torch.matmul(embeddings, embeddings.t())
        N = labels.size(0)
        mask_pos = labels.unsqueeze(1).eq(labels.unsqueeze(0))
        mask_neg = ~mask_pos
        # exclude diagonal
        mask_pos.fill_diagonal_(False)

        sp = sim[mask_pos].view(N, -1)
        sn = sim[mask_neg].view(N, -1)
        # from paper: alpha_p = relu(-sp + 1 + m), alpha_n = relu(sn + m)
        ap = torch.clamp_min(-sp + 1.0 + self.m, min=0.)
        an = torch.clamp_min(sn + self.m, min=0.)
        delta_p = 1 - self.m
        delta_n = self.m
        # logit for positive and negative
        logit_p = - self.gamma * ap * (sp - delta_p)
        logit_n = self.gamma * an * (sn - delta_n)
        # sum over each row (each sample)
        loss_p = torch.logsumexp(logit_p, dim=1)
        loss_n = torch.logsumexp(logit_n, dim=1)
        loss = torch.mean(F.softplus(loss_p + loss_n))
        return loss

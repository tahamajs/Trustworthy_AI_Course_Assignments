import torch


def fgsm_attack(model, x, y, epsilon, loss_fn=None, device='cuda'):
    was_training = model.training
    model.eval()
    x_adv = x.detach().clone().to(device)
    x_adv.requires_grad = True
    model.zero_grad()
    logits = model(x_adv)
    loss = loss_fn(logits, y) if loss_fn is not None else torch.nn.CrossEntropyLoss()(logits, y)
    loss.backward()
    pert = epsilon * x_adv.grad.sign()
    x_adv = (x_adv + pert).clamp(0, 1).detach()
    if was_training:
        model.train()
    return x_adv


def pgd_attack(model, x, y, epsilon, alpha, iters, loss_fn=None, device='cuda'):
    was_training = model.training
    model.eval()
    x_orig = x.detach().clone().to(device)
    x_adv = x_orig + torch.zeros_like(x_orig).uniform_(-epsilon, epsilon)
    x_adv = x_adv.clamp(0, 1).detach()
    for _ in range(iters):
        x_adv.requires_grad = True
        logits = model(x_adv)
        loss = loss_fn(logits, y) if loss_fn is not None else torch.nn.CrossEntropyLoss()(logits, y)
        model.zero_grad()
        loss.backward()
        grad = x_adv.grad.detach()
        x_adv = x_adv + alpha * grad.sign()
        x_adv = torch.max(torch.min(x_adv, x_orig + epsilon), x_orig - epsilon)
        x_adv = x_adv.clamp(0, 1).detach()
    if was_training:
        model.train()
    return x_adv

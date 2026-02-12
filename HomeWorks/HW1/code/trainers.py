# Optional: helper trainer classes (left minimal for clarity)
class Trainer:
    def __init__(self, model, optimizer, loss_fn, device='cuda'):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device

    def train_batch(self, x, y):
        x = x.to(self.device)
        y = y.to(self.device)
        self.optimizer.zero_grad()
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()

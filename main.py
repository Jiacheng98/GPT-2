import torch
# import torch._dynamo
# torch._dynamo.config.suppress_errors = True
import torch.nn as nn
from torch.nn import functional as F
import math
import tiktoken
import os

from data import DataLoaderLite
from model import GPTConfig, GPT
from early_stopper import EarlyStopper

log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")

torch.manual_seed(2)
device = "cpu"
if torch.cuda.is_available():
    torch.cuda.manual_seed(2)
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    torch.mps.manual_seed(2)
    device = "mps"
device = "cpu" # Force using cpu due to GPU memory constrain
with open(log_file, "w") as f:
    f.write(f"Using device: {device}\n")


def get_lr(it, max_lr, min_lr, warmup_steps, max_steps):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)


# Batch inference for the validation/testing data
def test(model, x, y, batch_size):
    model.eval()
    batch_no = x.shape[0] // batch_size
    loss = 0
    for each_batch in range(0, x.shape[0], batch_size):
        x_batch, y_batch = x[each_batch: each_batch + batch_size], \
        y[each_batch: each_batch + batch_size]
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        logits_batch, loss_batch = model(x_batch, y_batch)
        loss += 1/batch_no * loss_batch
    return loss

def main():
    # Get training data
    data_loader = DataLoaderLite(B=16, T=32)
    val_x, val_y, test_x, test_y = data_loader.get_val_test()

    # Model
    # model = GPT.from_pretrained('gpt2')
    model = GPT(GPTConfig())
    model.to(device)
    # model = torch.compile(model) # Doesn't work with Mac M1
    
    max_lr = 6e-4
    min_lr = max_lr * 0.1
    warmup_steps = 10
    max_steps = 50
    
    # Optimization, weight decay: penalizing large weights
    optimizer = torch.optim.AdamW(model.parameters(), lr = 3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    early_stopper = EarlyStopper(patience=50, min_delta=0.1)
    for step in range(5000):
        x, y = data_loader.next_batch()
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits, loss = model(x, y)
        loss.backward()
        # Gradient clipping: total norm upperbound: 1.0, prevent a shock gradient due to a bad batch
        # Return: Total norm (L2 norm, the root of the sum of the square of all parameters) of the parameter gradients (viewed as a single vector).
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # learning rate decay
        lr = get_lr(step, max_lr, min_lr, warmup_steps, max_steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()
        with open(log_file, "a") as f:
            f.write(f"Step {step}, training loss: {loss}, total_norm: {total_norm}, lr: {lr}\n")

        if step % 50 == 0:
            val_loss = test(model, val_x, val_y, data_loader.B)
            with open(log_file, "a") as f:
                f.write(f"Step {step}, validation loss: {val_loss}\n")
                # write model checkpoints
                checkpoint_path = os.path.join("log/", f"model_{step:05d}.pt")
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'config': model.config,
                    'step': step,
                    'val_loss': val_loss.item(),
                    'seed': 2
                }
                torch.save(checkpoint, checkpoint_path)

                if early_stopper.early_stop(val_loss): 
                    with open(log_file, "a") as f:
                        f.write("Early stopping!\n")            
                    break

    test_loss = test(model, test_x, test_y, data_loader.B)
    with open(log_file, "a") as f:
        f.write(f"Testing loss: {test_loss}\n")


if __name__ == "__main__":
    main()
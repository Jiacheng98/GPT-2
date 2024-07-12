import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import tiktoken

from data import DataLoaderLite
from model import GPTConfig, GPT

torch.manual_seed(2)
device = "cpu"
if torch.cuda.is_available():
    torch.cuda.manual_seed(2)
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    torch.mps.manual_seed(2)
    device = "mps"
print(f"using device: {device}")


def main():
    # Get training data
    train_loader = DataLoaderLite(B=4, T=32)

    # Model
    # model = GPT.from_pretrained('gpt2')
    model = GPT(GPTConfig())
    model.to(device)

    # Optimization
    optimizer = torch.optim.AdamW(model.parameters(), lr = 3e-4)
    for i in range(10000):
        x, y = train_loader.next_batch()
        optimizer.zero_grad()
        logits, loss = model(x.to(device), y.to(device))
        loss.backward()
        optimizer.step()
        print(f"Step {i}, loss: {loss}")


    num_return_sequences = 5
    max_length = 30

    # Prefix tokens
    enc = tiktoken.get_encoding('gpt2')
    tokens = enc.encode("Hello, I'm a happy student")
    tokens = torch.tensor(tokens, dtype = torch.long) 
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
    x = tokens.to(device)

    # Generate!
    while x.size(1) < max_length:
        with torch.no_grad():
            logits, _ = model(x) # (B, T, vocab_size)
            # Take the logits at the last position
            logits = logits[:, -1, :] # (B, vocab_size)
            probs = F.softmax(logits, dim = -1)
            # Do top-k sampling of 50
            topk_probs, topk_indices = torch.topk(probs, 50, dim = -1) # (B, 50)
            ix = torch.multinomial(topk_probs, 1) # (B, 1), select one token for each sequence
            xcol = torch.gather(topk_indices, -1, ix) # (B, 1), gather the corresponding indices
            x = torch.cat((x, xcol), dim = 1) # Append the selected token to the sequence

    # Print the generated text
    for i in range(num_return_sequences):
        tokens = x[i, :max_length].tolist()
        decoded = enc.decode(tokens)
        print(">", decoded)


if __name__ == "__main__":
    main()
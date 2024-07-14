import tiktoken
import torch
import os
from model import GPT
from torch.nn import functional as F

def generate(model):
    num_return_sequences = 5
    max_length = 50

    # Prefix tokens
    enc = tiktoken.get_encoding('gpt2')
    tokens = enc.encode("The course of true love never did run smooth.")
    tokens = torch.tensor(tokens, dtype = torch.long) 
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
    x = tokens

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
        print(f"> {decoded}\n")       

# Choose a model to complete sentences
checkpoint_path = os.path.join("log/model_00000.pt")
checkpoint = torch.load(checkpoint_path)
model = GPT(checkpoint["config"])
model.load_state_dict(checkpoint['model'])
generate(model)
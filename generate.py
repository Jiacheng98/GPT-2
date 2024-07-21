import tiktoken
import torch
import os
from model import GPT
from torch.nn import functional as F

def generate(model):
    num_sequences = 1
    max_length = 50
    # Prefix tokens
    enc = tiktoken.get_encoding('gpt2')
    tokens = enc.encode("The course of true love never did run smooth.")
    x = model.generate_sequence(model, tokens, num_sequences, max_length)
    
    # Print the generated text
    for i in range(num_sequences):
        tokens = x[i, :max_length].tolist()
        decoded = enc.decode(tokens)
        print(f"> {decoded}\n")       

# Choose a model to complete sentences
checkpoint_path = os.path.join("log/model_05000.pt")
checkpoint = torch.load(checkpoint_path)
model = GPT(checkpoint["config"])
model.load_state_dict(checkpoint['model'])
generate(model)
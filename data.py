import torch
import tiktoken

# Get batches of data
class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        with open('input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)

        train_tokens, val_tokens, test_tokens = tokens[:int(len(tokens) * 0.8)], \
        tokens[int(len(tokens) * 0.8):int(len(tokens) * 0.9)], tokens[int(len(tokens) * 0.9):]
        self.train_tokens, self.val_tokens, self.test_tokens = torch.tensor(train_tokens), torch.tensor(val_tokens), torch.tensor(test_tokens)
        print(f"Loaded {len(self.train_tokens)} train tokens, {len(self.val_tokens)} val tokens and {len(self.test_tokens)} test tokens.")
        print(f"1 epoch = {len(self.train_tokens) // (B * T)} batches")

        # state
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.train_tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T
        # if loading the next batch would be out of bounds, rest
        if self.current_position + (B * T + 1) > len(self.train_tokens):
            self.current_position = 0
        return x, y

    def get_val_test(self):
        T = self.T
        chunk_len = len(self.val_tokens) // T * T

        assert len(self.val_tokens) >= chunk_len + 1
        val_x = self.val_tokens[:chunk_len].view(len(self.val_tokens) // T, T) 
        val_y = self.val_tokens[1:chunk_len+1].view(len(self.val_tokens) // T, T) 

        assert len(self.test_tokens) >= chunk_len + 1
        test_x = self.test_tokens[:chunk_len].view(len(self.test_tokens) // T, T) 
        test_y = self.test_tokens[1:chunk_len+1].view(len(self.test_tokens) // T, T) 
        return val_x, val_y, test_x, test_y

# from bigram import encode, decode
import torch
with open("large.txt", 'r', encoding='utf-8') as f:
    text = f.read()
chars = list(sorted(set(list(text))))
vocab_size = len(chars)
stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

def adjust_tensor_to_block_size(tensor, BLOCK_SIZE, padding_value=1):
    """
    Adjusts the tensor to have a shape of (1, BLOCK_SIZE). Pads with padding_value or truncates the tensor as needed.
    """
    query_length = tensor.shape[1]
    
    if query_length < BLOCK_SIZE:
        # Pad the tensor
        padding_size = BLOCK_SIZE - query_length
        tensor = torch.cat([tensor, padding_value * torch.ones(1, padding_size)], dim=1)
    elif query_length > BLOCK_SIZE:
        # Truncate the tensor
        tensor = tensor[:, :BLOCK_SIZE]
    
    return tensor

#Tokenizer
def encode(string):
    out = []
    for char in string:
        out.append(stoi[char])
    return out

def decode(tokens):
    out = ""
    for token in tokens:
        out += itos[token]
    return out

query = input("Question:")
enc_query = adjust_tensor_to_block_size(torch.tensor([encode(query)]), 16)
print(enc_query)
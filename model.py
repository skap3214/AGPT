import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
#Hyper parameters
MODEL_PATH = './models/v1/AGPT.pth'
BATCH_SIZE = 32
BLOCK_SIZE = 50
EVAL_INTERVAL = 500
EPOCHS = 2000
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EVAL_ITERS = 200
MANUAL_SEED = 22
DATA = './datasets/large.txt'
N_EMBD = 100
HEADS = 20 #Should be divisible by N_EMBD
torch.manual_seed(MANUAL_SEED)

#Get dataset
with open(DATA, 'r', encoding='utf-8') as f:
    text = f.read()
chars = list(sorted(set(list(text))))
vocab_size = len(chars)
stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

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

#Train Test Split
data = torch.tensor(encode(text))
n = int(len(data)*0.9)
train_data = data[:n]
val_data = data[n:]

#For creating batches of size batch_size
def get_batch(split):
  data = train_data if split == "train" else val_data
  ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE, ))
  x = torch.stack([data[i:i+BLOCK_SIZE] for i in ix])
  y = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in ix])
  return x, y

def adjust_tensor_to_block_size(tensor, BLOCK_SIZE, padding_value=1):
    """
    Adjusts the tensor to have a shape of (1, BLOCK_SIZE). Pads with padding_value or truncates the tensor as needed.
    """
    query_length = tensor.shape[1]
    
    if query_length < BLOCK_SIZE:
        # Pad the tensor
        padding_size = BLOCK_SIZE - query_length
        tensor = torch.cat([tensor, padding_value * torch.ones(1, padding_size).long()], dim=1)
    elif query_length > BLOCK_SIZE:
        # Truncate the tensor
        tensor = tensor[:, :BLOCK_SIZE]
    
    return tensor


class FeedForward(nn.Module):
    def __init__(self, n_embd) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd),
            nn.ReLU()
        )
    def forward(self, x: torch.Tensor):
        return self.net(x)


class Head(nn.Module):
    '''
    My First Scaled Masked Self Attention Head!!!
    '''
    def __init__(self, head_size) -> None:
        super().__init__()
        self.key = nn.Linear(N_EMBD, (head_size), bias=False)
        self.query = nn.Linear(N_EMBD, (head_size), bias=False)
        self.value = nn.Linear(N_EMBD, (head_size), bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))# basically self.tril = torch.tril(...
    
    def forward(self, token_embd: torch.Tensor):
        B, T, C = token_embd.shape
        #Calculate keys, queries of sizes (B, T, head_size)
        k = self.key(token_embd)
        q = self.query(token_embd)

        #Create scaled masked attention matrix of size (T, T)
        weights = k @ q.transpose(-2, -1) * C**0.5
        weights = weights.masked_fill(self.tril == 0, float('-inf'))
        weights = torch.softmax(weights, dim=1)

        #Multiple weights with values(B, T, head_size) to get size (B, T, head_size)
        v = self.value(token_embd)
        output = weights @ v
        return output


class MultiHeadAttention(nn.Module):
    '''
    First Multi Head Attention!!!!!
    '''
    def __init__(self, num_heads, head_size) -> None:
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
    
    def forward(self, token_embd: torch.Tensor):
        #Concatenate outputs from all heads in the C dimension.
        #Each head corresponds to a different embddding dimension, thus will have differnet weights and stuff
        return torch.cat([head(token_embd) for head in self.heads], dim=-1)


class NotGPT(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(num_embeddings=vocab_size, embedding_dim=N_EMBD)
        self.positional_embeding_table = nn.Embedding(num_embeddings=BLOCK_SIZE, embedding_dim=N_EMBD)
        # self.sa_head = Head(N_EMBD)
        self.mh_head = MultiHeadAttention(HEADS, N_EMBD//HEADS)
        self.lm_head = nn.Linear(N_EMBD, vocab_size)
        self.ffwd = FeedForward(N_EMBD)

    def forward(self, idx: torch.Tensor, targets=None):
        B, T = idx.shape

        token_embd = self.token_embedding_table(idx) #Size: (B, T, N_EMBD)
        pos_embd = self.positional_embeding_table(torch.arange(T, device=DEVICE)) #Size: T, N_EMBD
        x = token_embd + pos_embd
        attended_x = self.ffwd(self.mh_head(x)) #Size: (B, T, N_EMBD)
        logits = self.lm_head(attended_x) #Size: (B, T, C)

        if targets == None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs,num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


model = NotGPT().to(DEVICE)
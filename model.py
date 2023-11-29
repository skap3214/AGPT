import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config as Config

torch.manual_seed(Config.MANUAL_SEED)

#Get dataset
with open(Config.DATA, 'r', encoding='utf-8') as f:
    text = f.read()
chars = sorted(set(list(text)))
vocab_size = len(chars)
itos = {i: ch for i, ch in enumerate(chars)}
stoi = {ch:i for i, ch in enumerate(chars)}
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
data = torch.tensor(encode(text)).to(Config.DEVICE)
n = int(len(data)*0.9)
train_data = data[:n]
val_data = data[n:]

#For creating batches of size batch_size
def get_batch(split):
  data = train_data if split == "train" else val_data
  ix = torch.randint(data.size(0) - Config.BLOCK_SIZE, (Config.BATCH_SIZE, ))
  x = torch.stack([data[i:i+Config.BLOCK_SIZE] for i in ix])
  y = torch.stack([data[i+1:i+Config.BLOCK_SIZE+1] for i in ix])
  return x, y

def adjust_tensor_to_block_size(tensor, BLOCK_SIZE, padding_value=1):
    """
    Adjusts the tensor to have a shape of (1, BLOCK_SIZE). Pads with padding_value or truncates the tensor as needed.
    """
    query_length = tensor.shape[1]
    
    if query_length < BLOCK_SIZE:
        # Pad the tensor
        padding_size = BLOCK_SIZE - query_length
        tensor = torch.cat([tensor, padding_value * torch.ones(1, padding_size, device=Config.DEVICE).long()], dim=1)
    elif query_length > BLOCK_SIZE:
        # Truncate the tensor
        tensor = tensor[:, :BLOCK_SIZE]
    
    return tensor


class FeedForward(nn.Module):
    def __init__(self, n_embd) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd*4),
            nn.ReLU(),
            nn.Linear(n_embd*4, n_embd),
            nn.Dropout(Config.DROPOUT)
        )
    def forward(self, x: torch.Tensor):
        return self.net(x)


class Head(nn.Module):
    '''
    My First Scaled Masked Self Attention Head!!!
    '''
    def __init__(self, head_size) -> None:
        super().__init__()
        self.key = nn.Linear(Config.N_EMBD, (head_size), bias=False)
        self.query = nn.Linear(Config.N_EMBD, (head_size), bias=False)
        self.value = nn.Linear(Config.N_EMBD, (head_size), bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(Config.BLOCK_SIZE, Config.BLOCK_SIZE)))# basically self.tril = torch.tril(...
        self.dropout = nn.Dropout(Config.DROPOUT)
    
    def forward(self, token_embd: torch.Tensor):
        B, T, C = token_embd.shape
        #Calculate keys, queries of sizes (B, T, head_size)
        k = self.key(token_embd)
        q = self.query(token_embd)

        #Create scaled masked attention matrix of size (T, T)
        weights = k @ q.transpose(-2, -1) * C**0.5
        weights = weights.masked_fill(self.tril == 0, float('-inf')) #If a value in self.tril == 0 then make the corresponding value in the weights matrix -inf
        weights = torch.softmax(weights, dim=1)
        weights = self.dropout(weights)

        #Multiple weights with values(B, T, head_size) to get size (B, T, head_size)
        v = self.value(token_embd)
        output = weights @ v
        return output

class ScaledHead(nn.Module):
    def __init__(self, n_embd) -> None:
        super().__init__()
        self.key = nn.Linear(n_embd, n_embd*2, bias=False)
        self.query = nn.Linear(n_embd, n_embd*2, bias=False)
        self.value = nn.Linear(n_embd, n_embd*2, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(Config.BLOCK_SIZE, Config.BLOCK_SIZE)))
        self.dropout = nn.Dropout(Config.DROPOUT)
        self.linear = nn.Linear(n_embd*2, n_embd)
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor):

        B, T, C = x.shape
        #Get key, query
        k = self.key(x) #(B, T, 2C)
        q = self.query(x)# (B, T, 2C)

        #Get attention matrix weights
        weights = k @ q.transpose(-2, -1) * C**0.5 #(T, T, 2C)
        weights = weights.masked_fill(self.tril == 0, float("-inf"))
        weights = torch.softmax(weights, dim=1)
        weights = self.dropout(weights)

        #Multiple attention matrix with value
        v = self.value(x) #(B, T, 2C)
        out = weights @ v
        out = self.linear(out) #(B, T, C)
        out = self.relu(out)
        return out


class MultiHeadAttention(nn.Module):
    '''
    First Multi Head Attention!!!!!
    '''
    def __init__(self, num_heads, head_size) -> None:
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(Config.N_EMBD, Config.N_EMBD)
        self.dropout = nn.Dropout(Config.DROPOUT)
    
    def forward(self, token_embd: torch.Tensor):
        #Concatenate outputs from all heads in the C dimension.
        #Each head corresponds to a different embddding dimension, thus will have differnet weights and stuff
        out = torch.cat([head(token_embd) for head in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class Block(nn.Module):
    def __init__(self, n_embd, num_head) -> None:
        assert n_embd % num_head == 0, f"not divisible: mod is {n_embd % num_head}"
        super().__init__()
        head_size = n_embd // num_head
        self.mh_head = MultiHeadAttention(num_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.layer_norm1 = nn.LayerNorm(n_embd)
        self.layer_norm2 = nn.LayerNorm(n_embd)
    
    def forward(self, x:torch.Tensor):
        out = x + self.mh_head(self.layer_norm1(x)) #Residual connections
        out = out + self.ffwd(self.layer_norm2(out))
        return out

class ScaledHeadBlock(nn.Module):
    def __init__(self, n_embd, num_head) -> None:
        #num_head is not used here nor is MultiHeadAttention
        super().__init__()
        self.sc_head = ScaledHead(n_embd)
        self.ffwd = FeedForward(n_embd)
        self.layer_norm1 = nn.LayerNorm(n_embd)
        self.layer_norm2 = nn.LayerNorm(n_embd)
    
    def forward(self, x:torch.Tensor):
        out = x + self.sc_head(self.layer_norm1(x))
        out = out + self.ffwd(self.layer_norm2(out))
        return out

class AGPT(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(num_embeddings=vocab_size, embedding_dim=Config.N_EMBD).to(Config.DEVICE)
        self.positional_embeding_table = nn.Embedding(num_embeddings=Config.BLOCK_SIZE, embedding_dim=Config.N_EMBD).to(Config.DEVICE)
        self.transformer_blocks = nn.Sequential(*[Block(Config.N_EMBD, Config.HEADS) for _ in range(Config.NUM_BLOCKS)]).to(Config.DEVICE)
        # self.scaled_transformer_block = nn.Sequential(*[ScaledHeadBlock(Config.N_EMBD, Config.HEADS) for _ in range(Config.NUM_BLOCKS)])
        self.layer_norm = nn.LayerNorm(Config.N_EMBD).to(Config.DEVICE)
        self.lm_head = nn.Linear(Config.N_EMBD, vocab_size, bias=False).to(Config.DEVICE)
        print(sum(p.numel() for p in self.parameters()), 'parameters')

    def forward(self, idx: torch.Tensor, targets=None):
        B, T = idx.shape
        assert T <= Config.BLOCK_SIZE, f"{T} is greater than {Config.BLOCK_SIZE}"
        token_embd = self.token_embedding_table(idx) #Size: (B, T, N_EMBD)
        pos_embd = self.positional_embeding_table(torch.arange(T, device=Config.DEVICE)) #Size: T, N_EMBD
        x = token_embd + pos_embd
        attended_x = self.transformer_blocks(x) #Size: (B, T, N_EMBD)
        # attended_x = self.scaled_transformer_block(x) #Size: (B, T, N_EMBD)
        logits = self.lm_head(self.layer_norm(attended_x)) #Size: (B, T, C)
        
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
            idx_cond = idx[:, -(Config.BLOCK_SIZE):]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs,num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

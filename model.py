import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config as Config
from modules import Head, MultiHeadAttention, FeedForward, ScaledHead
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
data = torch.tensor(encode(text))
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
        tensor = torch.cat([tensor, padding_value * torch.ones(1, padding_size).long()], dim=1)
    elif query_length > BLOCK_SIZE:
        # Truncate the tensor
        tensor = tensor[:, :BLOCK_SIZE]
    
    return tensor



class Block(nn.Module):
    def __init__(
            self, 
            n_embd, 
            num_head,
            block_size,
            dropout,
    ) -> None:
        assert n_embd % num_head == 0, f"not divisible: mod is {n_embd % num_head}"
        super().__init__()
        head_size = n_embd // num_head
        self.mh_head = MultiHeadAttention(head_size, n_embd, block_size, num_head, dropout)
        self.ffwd = FeedForward(n_embd)
        self.layer_norm1 = nn.LayerNorm(n_embd)
        self.layer_norm2 = nn.LayerNorm(n_embd)
    
    def forward(self, x:torch.Tensor):
        out = x + self.mh_head(self.layer_norm1(x)) #Residual connections
        out = out + self.ffwd(self.layer_norm2(out))
        return out

class ScaledHeadBlock(nn.Module):
    def __init__(
            self, 
            n_embd,
            block_size,
            dropout,
    ) -> None:
        #num_head is not used here nor is MultiHeadAttention
        super().__init__()
        self.sc_head = ScaledHead(n_embd, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.layer_norm1 = nn.LayerNorm(n_embd)
        self.layer_norm2 = nn.LayerNorm(n_embd)
    
    def forward(self, x:torch.Tensor):
        out = x + self.sc_head(self.layer_norm1(x))
        out = out + self.ffwd(self.layer_norm2(out))
        return out

class AGPT(nn.Module):

    def __init__(
            self,
            config: Config = Config,
    ):
        super().__init__()
        self.block_size = config.BLOCK_SIZE
        self.device = config.DEVICE
        self.token_embedding_table = nn.Embedding(num_embeddings=vocab_size, embedding_dim=config.N_EMBD)
        self.positional_embeding_table = nn.Embedding(num_embeddings=config.BLOCK_SIZE, embedding_dim=config.N_EMBD)
        self.transformer_blocks = nn.Sequential(*[Block(config.N_EMBD, config.HEADS, config.BLOCK_SIZE, config.DROPOUT) for _ in range(config.NUM_BLOCKS)])
        # self.scaled_transformer_block = nn.Sequential(*[ScaledHeadBlock(config.N_EMBD, config.HEADS) for _ in range(config.NUM_BLOCKS)])
        self.layer_norm = nn.LayerNorm(config.N_EMBD)
        self.lm_head = nn.Linear(config.N_EMBD, vocab_size, bias=False)
        print(sum(p.numel() for p in self.parameters()), 'parameters')

    def forward(self, idx: torch.Tensor, targets=None):
        B, T = idx.shape
        assert T <= self.BLOCK_SIZE, f"{T} is greater than {self.BLOCK_SIZE}"
        token_embd = self.token_embedding_table(idx) #Size: (B, T, N_EMBD)
        pos_embd = self.positional_embeding_table(torch.arange(T, device=self.DEVICE)) #Size: T, N_EMBD
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
            idx_cond = idx[:, -(self.BLOCK_SIZE):]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs,num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

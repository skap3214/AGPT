import torch
import torch.nn as nn
import torch.nn.functional as F
from config import SmallConfig as Config
from modules import Block
from tokenizer import train_tokenizer
torch.manual_seed(Config.MANUAL_SEED)
if torch.cuda.is_available():
    Config.DEVICE = "cuda"
else:
    Config.DEVICE = "cpu"

#Get dataset
tokenizer = train_tokenizer(Config.DATA, Config.MODEL_PATH, Config.BLOCK_SIZE)
vocab_size = tokenizer.get_vocab_size()
with open(Config.DATA, 'r', encoding='utf-8') as f:
    text = f.read()
# chars = sorted(set(list(text)))
# vocab_size = len(chars)
# itos = {i: ch for i, ch in enumerate(chars)}
# stoi = {ch:i for i, ch in enumerate(chars)}
#Tokenizer
def encode(string):
    return tokenizer.encode(string).ids

def decode(tokens):
    return tokenizer.decode(tokens, skip_special_tokens=False)

# Assume 'text' is your raw text data. If 'text' is a large single string,
# you need to split it into smaller strings (sentences or paragraphs) before encoding.
# For example, you could split the text into sentences:
text_sentences = text.split('.')  # This is a simplistic split. Consider using a more robust method.

# Now, encode each sentence and concatenate them
data = [encode(sentence) for sentence in text_sentences]
data = [token_id for sentence in data for token_id in sentence]  # Flatten the list of lists
data = torch.tensor(data)

print("Dataset shape:", data.shape)
n = int(len(data) * 0.9)
train_data = data[:n]
val_data = data[n:]

# Function for creating batches
def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - Config.BLOCK_SIZE, (Config.BATCH_SIZE, ))
    x = torch.stack([data[i:i + Config.BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i + 1:i + Config.BLOCK_SIZE + 1] for i in ix])
    return x, y

# Example use of get_batch

def adjust_tensor_to_block_size(tensor, BLOCK_SIZE, padding_value=0):
    """
    Adjusts the tensor to have a shape of (1, BLOCK_SIZE). Pads with padding_value or truncates the tensor as needed.
    """
    query_length = tensor.shape[1]
    
    if query_length < BLOCK_SIZE:
        # Pad the tensor
        padding_size = BLOCK_SIZE - query_length
        tensor = torch.cat([tensor, padding_value * torch.ones(1, padding_size, device=Config.DEVICE).long()], dim=1)
        tensor = torch.cat([tensor, padding_value * torch.ones(1, padding_size, device=Config.DEVICE).long()], dim=1)
    elif query_length > BLOCK_SIZE:
        # Truncate the tensor
        tensor = tensor[:, :BLOCK_SIZE]
    
    return tensor


Block = Block(Config.N_EMBD, Config.HEADS, Config.BLOCK_SIZE, Config.DROPOUT)

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
        self.transformer_blocks = nn.Sequential(*[Block for _ in range(config.NUM_BLOCKS)])
        # self.scaled_transformer_block = nn.Sequential(*[ScaledHeadBlock(config.N_EMBD, config.HEADS) for _ in range(config.NUM_BLOCKS)])
        self.layer_norm = nn.LayerNorm(config.N_EMBD)
        self.lm_head = nn.Linear(config.N_EMBD, vocab_size, bias=False)
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

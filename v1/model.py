import torch
import torch.nn as nn
import torch.nn.functional as F
from config import SmallConfig as Config
from modules import Block

torch.manual_seed(Config.MANUAL_SEED)
if torch.cuda.is_available():
    Config.DEVICE = "cuda"
else:
    Config.DEVICE = "cpu"


Block = Block(Config.N_EMBD, Config.HEADS, Config.BLOCK_SIZE, Config.DROPOUT)

class AGPT(nn.Module):

    def __init__(
            self,
            vocab_size: int,
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

import torch
import torch.nn as nn
import torch.nn.functional as F
from v4.config import SmallConfig as Config
from modules import RopeWithRMSNormBlock
torch.manual_seed(Config.MANUAL_SEED)
if torch.cuda.is_available():
    Config.DEVICE = "cuda"
else:
    Config.DEVICE = "cpu"

class AGPT(nn.Module):

    def __init__(
            self,
            vocab_size: int,
            config: Config = Config,
    ):
        super().__init__()
        self.block_size = config.BLOCK_SIZE
        self.device = config.DEVICE
        self.n_embd = config.N_EMBD
        self.token_embedding_table = nn.Embedding(num_embeddings=vocab_size, embedding_dim=config.N_EMBD)
        self.transformer_blocks = [RopeWithRMSNormBlock(config.N_EMBD, config.N_EMBD // config.HEADS, config.BLOCK_SIZE, config.DROPOUT).to(config.DEVICE) for _ in range(config.NUM_BLOCKS)]
        self.lm_head = nn.Linear(config.N_EMBD, vocab_size, bias=True)
        print(sum(p.numel() for p in self.parameters()), 'parameters')

    def init_weights(self) -> None:
        initrange = 0.1
        self.token_embedding_table.weight.data.uniform_(-initrange, initrange)
        self.lm_head.bias.data.zero_()
        self.lm_head.weight.data.uniform_(-initrange, initrange)
    
    def forward(self, idx: torch.Tensor, targets=None):
        B, T = idx.shape
        pad_token_id = 0
        pad_mask = (idx != pad_token_id)  # Creates a mask of shape (B, T)
        # pad_mask = pad_mask.to(self.device)
        assert T <= Config.BLOCK_SIZE, f"{T} is greater than {Config.BLOCK_SIZE}"
        token_embd = self.token_embedding_table(idx)#Size: (B, T, N_EMBD)
        attended_x = token_embd
        for block in self.transformer_blocks:
            attended_x = block(attended_x, pad_mask)
        logits = self.lm_head(attended_x) #Size: (B, T, C)
        if targets == None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens, streaming=False):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -(Config.BLOCK_SIZE):]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

            if streaming:
                yield idx_next
        if not streaming:
            return idx

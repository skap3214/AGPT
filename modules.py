'''
All nn.Modules that are implemented and used to create model
'''
import torch
from torch import nn

class Head(nn.Module):
    '''
    My First Scaled Masked Self Attention Head!!!
    '''
    def __init__(
            self, 
            head_size, #Head size to convert the embedding vector into
            n_embd, #Embedding dimension
            block_size, #context length of the model
            dropout, #amoutn of dropout to apply at the end
    ) -> None:
        super().__init__()
        self.key = nn.Linear(n_embd, (head_size), bias=False)
        self.query = nn.Linear(n_embd, (head_size), bias=False)
        self.value = nn.Linear(n_embd, (head_size), bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))# basically self.tril = torch.tril(...
        self.dropout = nn.Dropout(dropout)
    
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

        #Multiply weights with values(B, T, head_size) to get size (B, T, head_size)
        v = self.value(token_embd)
        output = weights @ v
        return output


class ScaledHead(nn.Module):
    def __init__(
            self, 
            n_embd,
            block_size,
            dropout,
    ) -> None:
        super().__init__()
        self.key = nn.Linear(n_embd, n_embd*2, bias=False)
        self.query = nn.Linear(n_embd, n_embd*2, bias=False)
        self.value = nn.Linear(n_embd, n_embd*2, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
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
    #TODO: make this and Head as one module
    '''
    def __init__(
            self, 
            head_size,
            n_embd,
            block_size,
            num_heads, 
            dropout
    ) -> None:
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, token_embd: torch.Tensor):
        #Concatenate outputs from all heads in the C dimension.
        #Each head corresponds to a different embddding dimension, thus will have differnet weights and stuff
        out = torch.cat([head(token_embd) for head in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    def __init__(
            self, 
            n_embd,
            dropout,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd*4),
            nn.ReLU(),
            nn.Linear(n_embd*4, n_embd),
            nn.Dropout(dropout)
        )
    def forward(self, x: torch.Tensor):
        return self.net(x)
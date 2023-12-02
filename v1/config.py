'''
Hyperparameters of the model
MODEL_PATH: path you want to save the model in
BATCH_SIZE: size of the batch you want to train the model with
BLOCK_SIZE: contex length of the model, how many tokens should it look at to predict the next token
EVAL_INTERVAL: when you want to print out the loss
EPOCHS: number of times you want to forward pass and back propogate to train the model
LR: learning rate at which the optimizer will update the weights
DEVICE: "cude" for gpu or "cpu" for cpu
EVAL_ITERS: how many times you want to iterate to find the validation loss
MANUAL_SEED: random seed to initialize the weights of the model
DATA: path to the dataset you want to use to train the model with
N_EMBD: embeddings model dimensions
NUM_BLOCKS: number of multi-head attention blocks in the decoder model
DROPOUT: between 0-1. rate at which the dropout will activate in the model
HEADS: number of self attention heads in each multi head attention model. This needs to be divisible by N_EMBD because 
each head holds equal number of dimensions for each logits.
'''

class Config:
    MODEL_PATH = 'models/v1/large/AGPT_large.pth'
    BATCH_SIZE = 64
    BLOCK_SIZE = 128
    EVAL_INTERVAL = 500
    EPOCHS = 5000
    LR = 2e-3
    DEVICE = "cuda"
    EVAL_ITERS = 200
    MANUAL_SEED = 22
    DATA = 'datasets/tiny_stories.txt'
    N_EMBD = 384
    NUM_BLOCKS = 6
    DROPOUT = 0.2
    HEADS = 8 

class MediumConfig:
    MODEL_PATH = 'models/v1/medium/AGPT_md.pth'
    BATCH_SIZE = 64
    BLOCK_SIZE = 160
    EVAL_INTERVAL = 500
    EPOCHS = 5000
    LR = 2e-3
    DEVICE = "cpu"
    EVAL_ITERS = 200
    MANUAL_SEED = 22
    DATA = 'datasets/qa_conversations.txt'
    N_EMBD = 128
    NUM_BLOCKS = 4
    DROPOUT = 0.1 
    HEADS = 8 

class SmallConfig:
    MODEL_PATH = 'models/v1/small/AGPT_small.pth'
    BATCH_SIZE = 32
    BLOCK_SIZE = 12
    EVAL_INTERVAL = 500
    EPOCHS = 3000
    LR = 1e-3
    DEVICE = "cpu"
    EVAL_ITERS = 200
    MANUAL_SEED = 22
    DATA = 'datasets/tiny_stories_small.txt'
    N_EMBD = 32
    NUM_BLOCKS = 3
    DROPOUT = 0.0
    HEADS = 4 

from dataclasses import dataclass

@dataclass
class Config:
    MODEL_PATH = './models/v1/AGPT_large.pth'
    BATCH_SIZE = 64
    BLOCK_SIZE = 256
    EVAL_INTERVAL = 500
    EPOCHS = 5000
    LR = 3e-3
    DEVICE = "cpu"
    EVAL_ITERS = 200
    MANUAL_SEED = 22
    DATA = './datasets/shakespeare.txt'
    N_EMBD = 384
    NUM_BLOCKS = 6
    DROPOUT = 0.2
    HEADS = 6 #Should be divisible by N_EMBD

class SmallConfig:
    MODEL_PATH = './models/v1/AGPT_small.pth'
    BATCH_SIZE = 32
    BLOCK_SIZE = 12 #Context Lenth of the model
    EVAL_INTERVAL = 500
    EPOCHS = 3000
    LR = 1e-3
    DEVICE = "cpu"
    EVAL_ITERS = 200
    MANUAL_SEED = 22
    DATA = './datasets/qa_conversations.txt'
    N_EMBD = 32
    NUM_BLOCKS = 3 #Number of Multi Attention Blocks
    DROPOUT = 0.0 #percentage of weights which will be randomly zeroed
    HEADS = 4 #Should be divisible by N_EMBD

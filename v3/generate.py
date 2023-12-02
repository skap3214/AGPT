import torch
from v3.model import AGPT, Config
from helpers import generate_response
from v3.tokenizer import get_tokenizer

tokenizer = get_tokenizer(Config.MODEL_PATH)

# Load model
model = AGPT(tokenizer.get_vocab_size()).to(Config.DEVICE)
model.load_state_dict(torch.load(Config.MODEL_PATH))
model.to(Config.DEVICE)
model.eval()


while True:
    query = input("Question: ")
    response = generate_response(
        query=query,
        model=model,
        tokenizer=tokenizer,
        block_size=Config.BLOCK_SIZE,
        device=Config.DEVICE
    )
    print("==========Model Response==========")
    print(response)

import torch
from model import AGPT, adjust_tensor_to_block_size, decode, encode, Config

# Load model
model = AGPT().to(Config.DEVICE)
model.load_state_dict(torch.load(Config.MODEL_PATH))
model.to(Config.DEVICE)
model.eval()

def generate_response(query):
    ten_query = torch.tensor([encode(query)], dtype=torch.long, device=Config.DEVICE)
    ten_query = ten_query.to(Config.DEVICE)
    enc_query = adjust_tensor_to_block_size(ten_query, Config.BLOCK_SIZE)
    enc_query = enc_query.to(Config.DEVICE)
    generated_tokens = model.generate(enc_query, 500)[0]
    return decode(generated_tokens.tolist())

if __name__ == "__main__":
    while True:
        query = input("Question: ")
        response = generate_response(query)
        print("==========Model Response==========")
        print(response)
        


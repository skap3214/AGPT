import torch
from model import NotGPT, adjust_tensor_to_block_size, decode, encode, BLOCK_SIZE, DEVICE

MODEL_PATH = 'bigram_language_model.pth'

# Load model
model = NotGPT().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

def generate_response(query):
    ten_query = torch.tensor([encode(query)], dtype=torch.long, device=DEVICE)
    enc_query = adjust_tensor_to_block_size(ten_query, BLOCK_SIZE)
    generated_tokens = model.generate(enc_query, 500)[0]
    return decode(generated_tokens.tolist())

if __name__ == "__main__":
    while True:
        query = input("Question: ")
        response = generate_response(query)
        print("==========Model Response==========")
        print(response)
        


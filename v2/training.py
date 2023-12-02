import torch
import json
import os
from v2.model import AGPT, Config
from time import time
from v2.tokenizer import train_tokenizer
# import wandb
# wandb.login()

#Get dataset
tokenizer = train_tokenizer(Config.DATA, Config.MODEL_PATH, Config.BLOCK_SIZE)
VOCAB_SIZE = tokenizer.get_vocab_size()
with open(Config.DATA, 'r', encoding='utf-8') as f:
    text = f.read()

#Tokenizer
def encode(string):
    return tokenizer.encode(string).ids

def decode(tokens):
    return tokenizer.decode(tokens, skip_special_tokens=True)

def split_on_token(text, token="endoftext"):
    """
    Split the text into chunks where each chunk ends with the specified token.
    
    :param text: The text to split.
    :param token: The token to split on.
    :return: A list of text chunks, each ending with the token.
    """

    chunks = text.split(token)
    chunks = [chunk + token for chunk in chunks if chunk.strip()]

    return chunks

text_sentences = split_on_token(text, "<|endoftext|>")

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

config = {
    'batch_size': Config.BATCH_SIZE,
    'block_size': Config.BLOCK_SIZE,
    'epochs': Config.EPOCHS,
    'lr': Config.LR,
    'n_embd': Config.N_EMBD,
    'manual_seed': Config.MANUAL_SEED,
    'data': Config.DATA
}
# run = wandb.init(
#     project='agpt_small',
#     job_type='train',
#     config=config,
# )
torch.manual_seed(Config.MANUAL_SEED)
model = AGPT(VOCAB_SIZE).to(Config.DEVICE)
# To store train and test loss
train_loss_list = []
test_loss_list = []

@torch.inference_mode()
def estimate_loss():
    final_loss = {}
    model.eval()
    for split in ["val"]:
        losses = torch.zeros(Config.EVAL_ITERS)
        for k in range(Config.EVAL_ITERS):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            # wandb.log({
            #     'val_loss': loss.item(),
            # })
            losses[k] = loss.item()
        final_loss[split] = losses.mean().item()
    model.train()
    return final_loss

optimizer = torch.optim.AdamW(params=model.parameters(), lr=Config.LR)

# Calculate the number of parameters
num_params = sum(p.numel() for p in model.parameters())

# Training Loop
start = time()
for iter in range(Config.EPOCHS):
    train_batch, test_batch = get_batch('train')
    train_batch = train_batch.to(Config.DEVICE)
    test_batch = test_batch.to(Config.DEVICE)
    logits, loss = model(train_batch, test_batch)
    # wandb.log({
    #     'loss': loss.item()
    # })
    train_loss_list.append(loss.item())
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if iter % Config.EVAL_INTERVAL == 0:
        losses = estimate_loss()
        test_loss_list.append(losses['val'])
        print(f"Epoch {iter} | Train Loss {train_loss_list[-1]} | Test Loss {losses['val']}")

TIME_TAKEN = time() - start
# Save the model
if not os.path.exists(os.path.dirname(Config.MODEL_PATH)):
    # If not, create the folder
    os.makedirs(os.path.dirname(Config.MODEL_PATH))
config_dict = {hyper: value for hyper, value in Config.__dict__.items() if not hyper.startswith("__")}
torch.save(model.state_dict(), Config.MODEL_PATH)

# Save additional details
metadata = config_dict | {
    'time_taken': TIME_TAKEN,
    'num_params': num_params,
    'train_loss_list': train_loss_list,
    'test_loss_list': test_loss_list
}

with open(f"{Config.MODEL_PATH}_details.json", 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"Model successfully trained!")
print(f"Parameters: {metadata['num_params']}")
print(f"Final Train Loss: {metadata['train_loss_list'][-1]}")
print(f"Final Test Loss: {metadata['test_loss_list'][-1]}")

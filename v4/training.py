import torch
import json
import os
from v4.model import AGPT, Config
from time import time
from v4.tokenizer import train_tokenizer
from helpers import split_on_token, get_batch, split_encode_dataset, create_torch_dataset

if Config.WANDB_LOG:
    import wandb
    wandb.login()

#Get dataset
tokenizer = train_tokenizer(Config.DATA, Config.MODEL_PATH, Config.BLOCK_SIZE)
VOCAB_SIZE = tokenizer.get_vocab_size()
print("VOCAB_SIZE", VOCAB_SIZE)
with open(Config.DATA, 'r', encoding='utf-8') as f:
    text = f.read()

splitted_dataset = split_on_token(text, "<|endoftext|>")

train_data, val_data = split_encode_dataset(splitted_dataset, tokenizer, 0.9)

train_loader = create_torch_dataset(train_data, Config.BLOCK_SIZE, Config.BATCH_SIZE)
val_loader = create_torch_dataset(val_data, Config.BLOCK_SIZE, Config.BATCH_SIZE)

config = {
    'batch_size': Config.BATCH_SIZE,
    'block_size': Config.BLOCK_SIZE,
    'epochs': Config.EPOCHS,
    'lr': Config.LR,
    'n_embd': Config.N_EMBD,
    'manual_seed': Config.MANUAL_SEED,
    'data': Config.DATA
}

_, version, size, name = Config.MODEL_PATH.split("/")
name = name.replace(".pth", "")
if Config.WANDB_LOG:
    run = wandb.init(
        project=f"{version}_{size}_{name}",
        job_type='train',
        config=config,
    )

torch.manual_seed(Config.MANUAL_SEED)
model = AGPT(VOCAB_SIZE).to(Config.DEVICE)
# To store train and test loss
train_loss_list = []
test_loss_list = []

@torch.inference_mode()
def estimate_loss():
    final_loss = {}
    model.eval()
    for split, loader in zip(["val"], [val_loader]):
        losses = torch.zeros(Config.EVAL_ITERS)
        for k, (x_val, y_val) in enumerate(loader):
            if k >= Config.EVAL_ITERS:
                break
            x_val, y_val = x_val.to(Config.DEVICE), y_val.to(Config.DEVICE)
            _, loss = model(x_val, y_val)
            if Config.WANDB_LOG:
                wandb.log({'val_loss': loss.item()})
            losses[k] = loss.item()
        final_loss[split] = losses.mean().item()
    model.train()
    return final_loss

optimizer = torch.optim.AdamW(params=model.parameters(), lr=Config.LR)

# Calculate the number of parameters
num_params = sum(p.numel() for p in model.parameters())

# Training Loop
start = time()
loops = 0
for iter in range(Config.EPOCHS):
    for i, (x_train, y_train) in enumerate(train_loader):
        loops += 1
        train_batch, test_batch = get_batch(train_data, Config.BLOCK_SIZE, Config.BATCH_SIZE)
        train_batch = train_batch.to(Config.DEVICE)
        test_batch = test_batch.to(Config.DEVICE)
        logits, loss = model(train_batch, test_batch)
        if Config.WANDB_LOG:
            wandb.log({
                'batch': loops,
                'loss': loss.item()
            })
        train_loss_list.append(loss.item())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        if i % Config.EVAL_INTERVAL == 0:
            losses = estimate_loss()
            test_loss_list.append(losses['val'])
            print(f"Epoch {iter+1} | Batch {i} Train Loss {train_loss_list[-1]} | Test Loss {losses['val']}")

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

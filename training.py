import torch
import json
import os
from model import AGPT, get_batch, Config
from time import time

torch.manual_seed(Config.MANUAL_SEED)
model = AGPT().to(Config.DEVICE)
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

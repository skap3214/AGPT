import torch
from torch.utils.data import Dataset, DataLoader
import tokenizers

def split_on_token(text, token="<|endoftext|>"):
    """
    Split the text into chunks where each chunk ends with the specified token.
    
    :param text: The text to split.
    :param token: The token to split on.
    :return: A list of text chunks, each ending with the token.
    """

    chunks = text.split(token)
    chunks = [chunk + token for chunk in chunks if chunk.strip()]

    return chunks

# Now, encode each sentence and concatenate them
def split_encode_dataset(dataset, tokenizer, train_split_size: float = 0.9) -> tuple:
    """
    Returns train_split, test_split
    """
    data = [tokenizer.encode(sentence).ids for sentence in dataset]
    data = [token_id for sentence in data for token_id in sentence]  # Flatten the list of lists
    data = torch.tensor(data)

    n = int(len(data) * train_split_size)
    train_data = data[:n]
    val_data = data[n:]

    return train_data, val_data

# Function for creating batches
def get_batch(data, block_size: int, batch_size: int):
    ix = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y

def create_torch_dataset(data, block_size: int, batch_size: int, shuffle=True) -> DataLoader:
    class TorchDataset(Dataset):
        def __init__(self, data, block_size):
            self.data = data
            self.block_size = block_size

        def __len__(self):
            # Total number of chunks available in the dataset
            return len(self.data) - self.block_size

        def __getitem__(self, idx):
            x = self.data[idx:idx + self.block_size]
            y = self.data[idx + 1:idx + self.block_size + 1]
            return x, y
    dataset = TorchDataset(data, block_size)
    dataloader = DataLoader(dataset, batch_size, shuffle)

    return dataloader

def generate_response(query, model, tokenizer: tokenizers.Tokenizer, block_size, device):
    ten_query = torch.tensor([tokenizer.encode(query).ids], dtype=torch.long, device=device)
    enc_query = ten_query.to(device)
    generated_tokens = model.generate(enc_query, block_size)[0]
    return tokenizer.decode(generated_tokens.tolist())

def generate_batched_response(query, model, tokenizer: tokenizers.Tokenizer, block_size, device):
    ten_query = torch.tensor([tokenizer.encode(query).ids], dtype=torch.long, device=device)
    enc_query = ten_query.to(device)
    generated_tokens = model.generate(enc_query, block_size)[0]
    return tokenizer.decode_batch(generated_tokens.tolist())

def generate_response_v2(query, model, tokenizer: tokenizers.Tokenizer, block_size, device, streaming=True):
    ten_query = torch.tensor([tokenizer.encode(query).ids], dtype=torch.long, device=device)
    enc_query = ten_query.to(device)
    
    if streaming:
        # Generate tokens in a streaming manner
        generated_tokens = []
        for token in model.generate(enc_query, block_size, streaming=True):
            yield token[0].cpu().tolist()[0]
    else:
        # Generate all tokens at once (non-streaming)
        generated_tokens = model.generate(enc_query, block_size)[0].cpu().tolist()
    
    return tokenizer.decode(generated_tokens)
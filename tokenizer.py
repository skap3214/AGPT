import os
import json
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from config import SmallConfig as Config

def train_tokenizer(data, path, max_length, eos_token="<|endoftext|>", special_tokens=[]):
    if os.path.exists((path + "/tokenizer.json")):
        print("Tokenizer already trained, using that one")
        return Tokenizer.from_file((path + "/tokenizer.json"))
    # Define your custom EOS token
    custom_eos_token = eos_token

    # Initialize a tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

    # Initialize a pre-tokenizer
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    # Initialize a trainer with special tokens, including your custom EOS and UNK tokens
    trainer = trainers.BpeTrainer(special_tokens=["[PAD]", "[UNK]", custom_eos_token]+special_tokens, show_progress=True)

    # Train the tokenizer
    tokenizer.train(files=[data], trainer=trainer)
    tokenizer.enable_padding(direction="right", pad_id=0, pad_token="[PAD]", length=max_length)
    tokenizer.enable_truncation(max_length)

    # Save the tokenizer
    path = os.path.dirname(path)
    if not os.path.exists(path):
        os.makedirs(path)
    final_path = (path + "/tokenizer.json")
    tokenizer.save(final_path)

    return tokenizer

# Tokenize some text
if __name__ == "__main__":
    custom_eos_token = "<|endoftext|>"
    tokenizer = train_tokenizer(Config.DATA, Config.MODEL_PATH, Config.BLOCK_SIZE)
    
    # Example text with your custom EOS token
    encoded = tokenizer.encode("Hello, y'all! How are you? " + custom_eos_token)
    
    # Check vocab size of tokenizer and output tokens
    print(tokenizer.get_vocab_size())
    print(encoded.tokens)
    print(type(tokenizer.decode(encoded.ids)))

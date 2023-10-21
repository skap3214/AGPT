import os
import json
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
import os
from config import SmallConfig
Config = SmallConfig

def train_tokenizer(data, path, max_length):
    
    # Initialize a tokenizer
    tokenizer = Tokenizer(models.BPE())

    # Initialize a pre-tokenizer
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    # Initialize a trainer with special tokens
    trainer = trainers.BpeTrainer(special_tokens=["[PAD]", "[UNK]"])

    # Train the tokenizer
    tokenizer.train(files=[data], trainer=trainer)
    tokenizer.enable_padding(direction="right", pad_id=0, pad_token="[PAD]", length=max_length)
    tokenizer.enable_truncation(max_length)

    # Save the tokenizer
    path = os.path.dirname(path)
    if not os.path.exists(path):
        # If not, create the folder
        os.makedirs(path)
    final_path = (path + "/tokenizer.json")
    tokenizer.save(final_path)

    return final_path
# Tokenize some text
if __name__ == "__main__":
    tokenizer_path = train_tokenizer(Config.DATA, Config.MODEL_PATH)
    tokenizer = Tokenizer.from_file(tokenizer_path)
    encoded = tokenizer.encode("Hello, y'all! How are you?")
    #Check vocab size of tokenizer
    print(tokenizer.get_vocab_size())
    print(encoded.tokens)
import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from config import SmallConfig
Config = SmallConfig

def train_tokenizer(data, path):
    # Initialize a tokenizer
    tokenizer = Tokenizer(BPE())

    # Initialize a pre-tokenizer
    tokenizer.pre_tokenizer = Whitespace()

    # Initialize a trainer
    trainer = BpeTrainer()

    # Train the tokenizer
    tokenizer.train(files=[data], trainer=trainer)

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
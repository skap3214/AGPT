import datetime
from colorama import init, Fore, Style
import textwrap
import sys
import time
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

init()

def print_with_color(text, color):
    print(color + text + Style.RESET_ALL)

while True:
    query = input(Fore.CYAN + "Your Question (Type 'exit' to quit): " + Style.RESET_ALL)
    
    if query.lower() == 'exit':
        break

    try:
        print(Fore.YELLOW + "Generating response, please wait..." + Style.RESET_ALL)
        
        response = generate_response(
            query=query,
            model=model,
            tokenizer=tokenizer,
            block_size=Config.BLOCK_SIZE,
            device=Config.DEVICE
        )

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print_with_color(f"\n========== Model Response at {timestamp} ==========", Fore.GREEN)
        print(textwrap.fill(response, width=70))

    except Exception as e:
        print_with_color("An error occurred: " + str(e), Fore.RED)
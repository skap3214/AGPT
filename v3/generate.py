import sys
sys.path.append("")
import time
import tqdm
import datetime
from colorama import init, Fore, Style, Back
import textwrap
import torch
import os
from v3.model import AGPT, Config
from helpers import generate_response
from v3.tokenizer import get_tokenizer

if torch.cuda.is_available():
    Config.DEVICE = "cuda"
else:
    Config.DEVICE = "cpu"

tokenizer = get_tokenizer(Config.MODEL_PATH)

# Load model
model = AGPT(tokenizer.get_vocab_size()).to(Config.DEVICE)
model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=torch.device(Config.DEVICE)))
model.to(Config.DEVICE)
model.eval()

# Initialize colorama
init(autoreset=True)

def clear_screen():
    """ Clears the screen based on the operating system. """
    os.system('cls' if os.name == 'nt' else 'clear')

def print_with_color(text, color, background=Back.BLACK, style=Style.NORMAL, end="\n"):
    print(style + background + color + text + Style.RESET_ALL, end=end)

def print_header():
    """ Prints the chat header. """
    header_text = " Welcome to YouLearn Chat Interface "
    print_with_color("+" + "-" * (len(header_text) + 2) + "+", Fore.BLUE)
    print_with_color(f"|{header_text}|", Fore.CYAN, Back.BLUE, Style.BRIGHT)
    print_with_color("+" + "-" * (len(header_text) + 2) + "+", Fore.BLUE)

def print_footer():
    """ Prints the chat footer. """
    footer_text = " Thank you for using YouLearn Chat! "
    print_with_color("+" + "-" * (len(footer_text) + 2) + "+", Fore.BLUE)
    print_with_color(f"|{footer_text}|", Fore.CYAN, Back.BLUE, Style.BRIGHT)
    print_with_color("+" + "-" * (len(footer_text) + 2) + "+", Fore.BLUE)

def format_timestamp():
    """ Returns formatted current timestamp. """
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def simulate_typing_effect(text):
    """ Simulates a typing effect for the given text. """
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(0.05)

def print_response_box(response):
    """ Prints the response in a box. """
    lines = textwrap.wrap(response, width=70)
    width = max(len(line) for line in lines)
    print_with_color("+" + "-" * (width + 2) + "+", Fore.GREEN)
    for line in lines:
        print_with_color(f"| {line.ljust(width)} |", Fore.GREEN)
    print_with_color("+" + "-" * (width + 2) + "+", Fore.GREEN)

def main():
    clear_screen()
    print_header()

    while True:
        query = input(Fore.YELLOW + Style.BRIGHT + "Your Question (Type 'exit' to quit): " + Style.RESET_ALL)

        if query.lower() == 'exit':
            break

        try:
            print_with_color("Generating response, please wait...", Fore.YELLOW, style=Style.DIM)
            
            # Placeholder for the actual response generation
            response = generate_response(
                query=query,
                model=model,
                tokenizer=tokenizer,
                block_size=Config.BLOCK_SIZE,
                device=Config.DEVICE
            )

            timestamp = format_timestamp()
            print_with_color(f"\n--- Model Response at {timestamp} ---", Fore.GREEN, style=Style.BRIGHT)
            simulate_typing_effect(response)
            print("\n")
            print_response_box(response)

        except Exception as e:
            print_with_color("An error occurred: " + str(e), Fore.RED, style=Style.BRIGHT)

    print_footer()

if __name__ == "__main__":
    main()
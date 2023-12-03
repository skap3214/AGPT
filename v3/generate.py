import sys
sys.path.append("")
import datetime
import torch
import os
from v3.model import AGPT, Config
from helpers import generate_response
from v3.tokenizer import get_tokenizer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

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


console = Console()

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def dynamic_greeting():
    current_hour = datetime.datetime.now().hour
    if 5 <= current_hour < 12:
        return "Good Morning!"
    elif 12 <= current_hour < 18:
        return "Good Afternoon!"
    else:
        return "Good Evening!"

def print_header():
    greeting = dynamic_greeting()
    console.print(f"{greeting} Welcome to YouLearn Chat Interface", style="bold blue")

def print_footer():
    console.print("Thank you for using YouLearn Chat! Have a great day!", style="bold blue")

def print_help():
    table = Table(title="Chat Commands")
    table.add_column("Command", style="dim", width=12)
    table.add_column("Description", style="dim")
    table.add_row("exit", "Exit the chat interface.")
    table.add_row("help", "Display this help message.")
    console.print(table)

def format_timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def print_response_box(response):
    response_panel = Panel(response, title="Response", border_style="green")
    console.print(response_panel)

def main():
    clear_screen()
    print_header()

    while True:
        query = console.input("[bold yellow]Your Question (Type 'exit' or 'help' for commands): ")

        if query.lower() == 'exit':
            break
        elif query.lower() == 'help':
            print_help()
            continue

        try:
            with console.status("[bold yellow]Processing your request...", spinner="dots"):
                # Simulate response generation
                response = generate_response(
                query=query,
                model=model,
                tokenizer=tokenizer,
                block_size=Config.BLOCK_SIZE,
                device=Config.DEVICE
                )
                
                timestamp = format_timestamp()
                console.print(f"--- Model Response at {timestamp} ---", style="bold green")
                print_response_box(response)
        except Exception as e:
            console.print(f"An error occurred: {e}", style="bold red")

    print_footer()

if __name__ == "__main__":
    main()
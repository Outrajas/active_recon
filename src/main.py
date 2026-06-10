import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.style_encoder import get_available_styles, validate_style
from src.inference import run_rewrite
from src.pdf_ingestor import process_new_pdfs
from src.dataset_builder import build_dataset_incremental
import subprocess

def print_banner():
    print("\n" + "="*40)
    print("JYADA AALAS")
    print("="*40)

def list_styles():
    print("\nAvailable styles:")
    for i, style in enumerate(get_available_styles(), 1):
        print(f"  {i}. {style}")

def rewrite_flow():
    list_styles()
    choice = input("\nChoose style (number or name): ").strip()
    styles = get_available_styles()
    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(styles):
            style = styles[idx]
        else:
            print("Invalid number.")
            return
    else:
        style = choice.upper()
        if not validate_style(style):
            print(f"Invalid style. Choose from: {', '.join(styles)}")
            return
    print(f"\nSelected style: {style}")
    print("Enter text to rewrite (press Enter twice to finish):")
    lines = []
    while True:
        line = input()
        if line == "" and lines and lines[-1] == "":
            break
        lines.append(line)
    text = "\n".join(lines).strip()
    if not text:
        print("No text provided.")
        return
    print("\n--- Original ---")
    print(text)
    print("\n--- Rewriting... ---")
    result = run_rewrite(text, style)
    print("\n--- Rewritten ---")
    print(result)

def process_pdfs_flow():
    print("\nProcessing new PDFs (skipping already processed)...")
    process_new_pdfs()
    input("\nPress Enter to continue...")

def build_dataset_flow():
    print("\nBuilding dataset incrementally (new chunks only)...")
    build_dataset_incremental()
    input("\nPress Enter to continue...")

def train_flow():
    print("\nStarting incremental training (resuming from existing LoRA if any)...")
    subprocess.run([sys.executable, "src/train.py"])
    input("\nPress Enter to continue...")

def main():
    while True:
        print_banner()
        print("\n1. Rewrite Text")
        print("2. List Styles")
        print("3. Process New PDFs")
        print("4. Build/Update Dataset")
        print("5. Train Model (incremental)")
        print("6. Exit")
        option = input("\nChoose option: ").strip()
        if option == "1":
            rewrite_flow()
        elif option == "2":
            list_styles()
        elif option == "3":
            process_pdfs_flow()
        elif option == "4":
            build_dataset_flow()
        elif option == "5":
            train_flow()
        elif option == "6":
            print("Goodbye.")
            break
        else:
            print("Invalid option. Try again.")

if __name__ == "__main__":
    main()


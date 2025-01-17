# gui.py
import tkinter as tk
from tkinter import scrolledtext
from model_utils import load_models, predict
import csv
import os

# Path to the dataset file
DATASET_FILE = "grammar_correction_dataset.csv"

# Initialize the dataset file if not exists
if not os.path.exists(DATASET_FILE):
    with open(DATASET_FILE, mode="w", newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Original Text", "Corrected Text"])

# Load models
root = tk.Tk()
root.title("Tamil Grammar Checker")
model, vectorizer, encoder = load_models()

def check_grammar():
    input_text = input_text_box.get("1.0", tk.END).strip()
    if input_text:
        corrected_text = predict(input_text, model, vectorizer, encoder)
        
        # Display the corrected text in the result box
        result_text_box.config(state='normal')
        result_text_box.delete('1.0', tk.END)
        result_text_box.insert(tk.END, corrected_text)
        result_text_box.config(state='disabled')
        
        # Append to dataset
        with open(DATASET_FILE, mode="a", newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([input_text, corrected_text])

# UI components
input_text_box = scrolledtext.ScrolledText(root, height=10, width=50)
input_text_box.pack(pady=10)

check_button = tk.Button(root, text="Check Grammar", command=check_grammar)
check_button.pack(pady=5)

result_text_box = scrolledtext.ScrolledText(root, height=10, width=50)
result_text_box.pack(pady=10)
result_text_box.config(state='disabled')

root.mainloop()

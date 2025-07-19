# flan-t5-large-qlora-chatbot
## Description
An automated pipeline extracts and cleans legal texts (selectable PDFs via PyPDF2/pdfplumber and scanned documents via Tesseract), generates high‑quality question–answer pairs (Barthez + cross‑encoder ranking for questions; QAmembert or T5 summarizer for answers), applies filters (grammar checks, semantic overlap validation, duplicate removal), and then fine‑tunes Flan‑T5‑large in 8‑bit using QLoRA.

## Prerequisites
Python 3.8+

CUDA support for PyTorch

Libraries: transformers, datasets, peft, bitsandbytes, evaluate, nltk, PyPDF2, pdfplumber, pytesseract, Grammalecte, LanguageTool

Data file: dataset_full.json (question/answer pairs in JSON format)

## Installation
### 1. Clone the repository

```bash
git https://github.com/abdouelouatiki/flan-t5-large-qlora-chatbot
cd flan-t5-large-qlora-chatbot
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage
Make the training script executable and run it:

```bash
chmod +x train_flan_t5_qlora.py
./train_flan_t5_qlora.p
```
Training checkpoints and logs will be saved under ./flan_t5_large_qlora_optim.

## Training Script Details
Model: google/flan-t5-large quantized to 8‑bit with BitsAndBytes + QLoRA (r=16, α=64, dropout=0.0)

Batch: per‑device batch size = 6, gradient accumulation steps = 2 (≈6 GB VRAM)

## Hyperparameters:

Learning rate = 3 × 10⁻⁴

Max epochs = 8, EarlyStopping patience = 3

Scheduler = cosine with warmup ratio = 0.10

Optimizer = AdamW

Weight decay = 0.01, max gradient norm = 1.0

Label smoothing = 0.1

Generation: beams = 4, max generation length = 200

Miscellaneous: gradient checkpointing enabled; no fp16 or bf16.

## Experimental Results
Eval Loss: decreased from 2.35 to 2.20 over 8 epochs

ROUGE‑1/2/L (%): reached 30.18 / 19.38 / 26.97 at epoch 8 (peak ROUGE‑1 of 30.05 at epoch 5)

## License
This project is licensed under the MIT License. See the LICENSE file for details.

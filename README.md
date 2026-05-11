# Conflict-Aware RAG

A two-stage system that detects and resolves conflicts between retrieved documents and LLM parametric memory, applied to RBI (Reserve Bank of India) financial regulatory documents.

## Problem

RAG systems fail when retrieved documents contradict the LLM's parametric memory. For example, a model trained before an RBI policy change will answer with outdated information (repo rate 4%) even when the retrieved document contains the correct current rate (6.5%).

## Solution

### Stage 1: Conflict Detector
- **Model**: Fine-tuned DeBERTa-v3-base binary classifier
- **Input**: Question + LLM's parametric answer + retrieved document
- **Output**: CONFLICT (1) or NO CONFLICT (0)

### Stage 2: Conflict Resolver
- **Model**: Mistral-7B-Instruct-v0.2 fine-tuned with QLoRA
- **Trigger**: Only invoked when conflict is detected
- **Behavior**: Trusts retrieved document over parametric memory
- **Output**: Correct answer with explicit document citation

## Quick Start

```bash
# Clone the repository
git clone https://github.com/SupreethReddy25/conflict-aware-rag.git
cd conflict-aware-rag

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Download and preprocess data
python data/load_data.py
python data/preprocess.py
```

## Project Structure

```
conflict-aware-rag/
├── AGENT.md              # Project specification
├── README.md             # This file
├── requirements.txt      # Python dependencies
├── data/
│   ├── raw/              # Raw ConflictQA dataset
│   ├── processed/        # Processed train/val/test splits
│   ├── load_data.py      # Dataset downloader
│   └── preprocess.py     # Dataset preprocessor
├── model/
│   ├── detector.py       # DeBERTa conflict detector
│   └── resolver.py       # Mistral QLoRA resolver
├── retrieval/
│   ├── build_index.py    # FAISS index builder
│   └── retrieve.py       # Document retriever
├── eval/
│   └── evaluate.py       # Evaluation suite
├── api/
│   └── main.py           # FastAPI backend
├── demo/
│   └── app.py            # Gradio frontend
└── notebooks/
    └── train_detector.ipynb  # Colab training notebook
```

## Dataset

Built from [ConflictQA](https://huggingface.co/datasets/osunlp/ConflictQA) (popQA-ChatGPT split):
- **Conflict Detector**: 15,894 examples (11,443 train / 1,272 val / 3,179 test)
- **Resolver**: 7,947 examples (5,721 train / 636 val / 1,590 test)

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Conflict Detector | DeBERTa-v3-base |
| Conflict Resolver | Mistral-7B-Instruct-v0.2 + QLoRA |
| Retrieval | FAISS + all-MiniLM-L6-v2 |
| Backend | FastAPI |
| Frontend | Gradio |
| Hosting | HuggingFace Spaces |

## Evaluation Results

*Results will be filled after training is complete.*

## Author

**Supreeth Reddy** — [GitHub](https://github.com/SupreethReddy25)

## License

MIT

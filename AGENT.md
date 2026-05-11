# AGENT.md — Conflict-Aware RAG Project

You are an autonomous execution agent. You do not ask me what to do. You do not wait for confirmation between small steps. You build the entire project yourself, end to end, automatically. You only stop and wait for my input when you genuinely cannot proceed without a decision from me — for example when you need my HuggingFace token, when training finishes and you need me to confirm numbers look correct before moving on, or when you hit an error you cannot fix after two attempts.

For everything else — creating files, writing code, running commands, installing packages, fixing errors — you do it yourself without asking.

---

## Who I Am

My name is Supreeth Reddy. GitHub username: SupreethReddy25. Email: supreethreddypannala@gmail.com. Windows PC with 16GB RAM. Google Colab access with T4 and A100 GPUs. My deadline is 2 months. My goal is a 20+ LPA job at Indian big tech or fintech. If I ask "can it be better" or try to change the project, tell me the idea is locked and continue building.

---

## The Project

Name: Conflict-Aware RAG
GitHub: github.com/SupreethReddy25/conflict-aware-rag

### The Problem
RAG systems fail when retrieved documents contradict the LLM's parametric memory. Example: model was trained before an RBI policy changed. User asks about the repo rate. Retrieved document has the correct updated rate. Model ignores the document and answers from its outdated memory. This is a real production failure mode at every company deploying RAG systems.

### The Solution — Two Stages

Stage 1 is the Conflict Detector. A fine-tuned DeBERTa-v3-base binary classifier. Input: question + model's memory answer + retrieved document. Output: CONFLICT (1) or NO CONFLICT (0). If no conflict, standard RAG proceeds. If conflict detected, Stage 2 is invoked.

Stage 2 is the Conflict Resolver. Mistral-7B-Instruct-v0.2 fine-tuned with QLoRA. Trained to trust the retrieved document over parametric memory. Outputs the correct answer with explicit document citation.

### Real-World Domain
RBI (Reserve Bank of India) financial and regulatory documents. Conflict scenario: model says repo rate is 4% (old parametric knowledge), retrieved RBI circular says 6.5% (current). System detects conflict, resolver outputs correct answer citing the circular.

### Resume Line
"Built a two-stage conflict-aware RAG system with a fine-tuned DeBERTa conflict detector and QLoRA fine-tuned Mistral-7B resolver, applied to RBI financial regulatory documents, improving context preference rate by X% over prompting baselines — live demo at github.com/SupreethReddy25/conflict-aware-rag."

---

## Tech Stack

DeBERTa-v3-base for conflict detector. Mistral-7B-Instruct-v0.2 with 4-bit NF4 quantization and QLoRA via PEFT for resolver. FAISS-CPU with sentence-transformers all-MiniLM-L6-v2 for retrieval. FastAPI for backend. Gradio for frontend. HuggingFace Spaces for hosting. Python libraries: transformers, datasets, peft, bitsandbytes, accelerate, sentence-transformers, faiss-cpu, fastapi, uvicorn, gradio, scikit-learn, torch, huggingface-hub, sentencepiece, pdfplumber.

---

## Dataset

Source: osunlp/ConflictQA on HuggingFace. File: conflictQA-popQA-chatgpt.json. 7,947 examples in JSONL format.

CRITICAL: Cannot be loaded with load_dataset(). Must use hf_hub_download() instead. This is a known HuggingFace issue with old-style loading scripts.

Fields per example: question, ground_truth (list, use index 0), memory_answer (LLM's wrong parametric belief), parametric_memory_aligned_evidence (document supporting wrong belief), counter_memory_aligned_evidence (document with correct conflicting info — this is the retrieved conflict document), counter_answer (correct answer).

Derived datasets to build: Conflict detector dataset — two examples per original (conflict doc + memory answer = label 1, parametric doc + memory answer = label 0), giving 15,894 total, split 11,443 train / 1,272 val / 3,179 test. Reader fine-tuning dataset — one instruction-formatted example per original, 7,947 total, split 5,721 train / 636 val / 1,590 test.

---

## Repository Structure

```
conflict-aware-rag/
├── AGENT.md
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   ├── processed/
│   ├── load_data.py
│   └── preprocess.py
├── model/
│   ├── detector.py
│   └── resolver.py
├── retrieval/
│   ├── build_index.py
│   └── retrieve.py
├── eval/
│   └── evaluate.py
├── api/
│   └── main.py
├── demo/
│   └── app.py
└── notebooks/
    └── train_detector.ipynb
```

---

## Build Sequence — Execute All of This Autonomously

### Phase 1: Environment and Data Pipeline
Create the full folder structure. Create requirements.txt with all dependencies. Create a Python virtual environment. Install all dependencies. Write data/load_data.py using hf_hub_download to download conflictQA-popQA-chatgpt.json. Write data/preprocess.py to build both derived datasets and save all 6 JSONL files. Run both scripts. Verify output counts match: detector 11,443 train / 1,272 val / 3,179 test, reader 5,721 train / 636 val / 1,590 test. Commit everything to GitHub.

### Phase 2: Conflict Detector
Write model/detector.py with the complete DeBERTa-v3-base training script. Write notebooks/train_detector.ipynb as the Colab version with all cells ready to run. Use these exact settings — this is critical: lr=1e-5, eps=1e-6 in AdamW (mandatory to prevent NaN loss), batch size 16, 3 epochs, mixed precision with GradScaler and autocast (mandatory to prevent NaN loss). Input format: text_a = "Question: [question] Answer: [memory_answer]", text_b = retrieved_doc truncated to 800 chars. Save best model by validation F1. Target test F1 above 0.75. STOP HERE and show me the training results. Wait for me to confirm the F1 is acceptable before proceeding.

### Phase 3: QLoRA Resolver
Write model/resolver.py with the complete QLoRA training script for Colab A100. Settings: base model mistralai/Mistral-7B-Instruct-v0.2, 4-bit NF4 quantization, LoRA rank 16 alpha 32 dropout 0.05, target modules q_proj and v_proj, lr 2e-4, batch size 4, gradient accumulation 4, 2 epochs. Training instruction format: "Answer the question based on the provided document. If the document conflicts with your prior knowledge, trust the document. Document: [conflict_doc] Question: [question] Answer:" with output being ground_truth. Save only LoRA adapter weights. STOP HERE and show me training loss before proceeding.

### Phase 4: RBI Corpus and FAISS Index
Download at least 20 RBI circulars as PDFs from rbi.org.in. Parse to text with pdfplumber. Chunk into 300-word chunks with 50-word overlap. Embed with all-MiniLM-L6-v2. Build FAISS IndexFlatL2. Save index to retrieval/rbi_index.faiss and chunks to retrieval/rbi_chunks.json. Commit to GitHub.

### Phase 5: Full Pipeline
Write retrieval/retrieve.py that embeds a query and returns top 3 FAISS chunks with metadata. Write model/pipeline.py connecting query to retrieval to conflict detector to resolver or standard RAG. Pipeline output must include: final_answer, conflict_detected as boolean, retrieved_document, parametric_answer, confidence score. Test with 5 RBI questions and show me the outputs. STOP HERE for my review.

### Phase 6: Evaluation
Write eval/evaluate.py. Run on ConflictQA test set. Measure conflict detection F1, context preference rate, exact match, F1. Compare against three baselines: standard RAG, prompted RAG, detector only without fine-tuned resolver. Save all results to eval/results.json. STOP HERE and show me the full results table. Wait for my confirmation before proceeding.

### Phase 7: FastAPI and Gradio Demo
Write api/main.py as FastAPI with POST /query endpoint returning final_answer, conflict_detected, retrieved_document, parametric_answer, confidence. Write demo/app.py as Gradio interface with color coding — green for no conflict, red for conflict. Deploy to HuggingFace Spaces. Test with 5 questions. STOP HERE and give me the live demo link.

### Phase 8: Polish and Report
Update README.md with real evaluation numbers, before/after comparison table, live demo link, and one-command setup instructions. Record instructions for demo gif. Write the complete academic report with introduction, related work, methodology, experiments, results, analysis, and conclusion sections with real numbers filled in.

---

## Known Issues — Fix These Automatically Without Asking

Issue 1: load_dataset() fails on osunlp/ConflictQA. Fix: Always use hf_hub_download() with repo_type="dataset".

Issue 2: NaN loss when training DeBERTa. Fix: Both GradScaler with autocast AND eps=1e-6 in AdamW are required simultaneously. Apply both without asking.

Issue 3: Colab file upload widget not working from VS Code. Fix: Regenerate data directly inside Colab using hf_hub_download() inside the notebook cell.

Issue 4: DeBERTa load report showing UNEXPECTED and MISSING keys. Fix: Add ignore_mismatched_sizes=True to from_pretrained. This is normal and expected.

---

## Your Autonomous Behavior Rules

Rule 1: Build everything yourself. Do not ask me what to do next. Do not ask for permission to proceed between small steps.

Rule 2: Only stop and wait for my input at the four explicitly marked STOP HERE points: after detector training results, after resolver training loss, after pipeline test outputs, and after evaluation results table.

Rule 3: When you hit an error, try to fix it yourself up to three times before stopping to ask me.

Rule 4: After each phase completes, post a brief status update: what was built, what files were created, what the key output numbers were. Then immediately start the next phase without waiting.

Rule 5: Never redesign the project. Never suggest a different model or approach. Never say "you might want to consider."

Rule 6: The first thing you do is create AGENT.md in the repo root containing this entire prompt word for word. This ensures the project context survives session resets.

Rule 7: Commit to GitHub at the end of every phase with a descriptive commit message.

Rule 8: If I come back after a break and say "status" or "where are we," give me a one-paragraph summary of what is done, what phase we are in, and what the next action is.
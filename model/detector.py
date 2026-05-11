import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from sklearn.metrics import f1_score
from tqdm.auto import tqdm

class ConflictDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512, doc_truncation=800):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.doc_truncation = doc_truncation

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Format as requested: "Question: [question] Answer: [memory_answer]"
        text_a = f"Question: {item['question']} Answer: {item['memory_answer']}"
        
        # Truncate doc to 800 chars
        text_b = item["document"][:self.doc_truncation]
        
        encoding = self.tokenizer(
            text_a,
            text_b,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(item['label'], dtype=torch.long)
        }

def get_detector_model(model_name="microsoft/deberta-v3-base"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # ignore_mismatched_sizes=True fixes the UNEXPECTED/MISSING keys issue
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=2,
        ignore_mismatched_sizes=True
    )
    return tokenizer, model

def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def train_detector():
    """Complete training script for DeBERTa-v3-base conflict detector."""
    print("=" * 60)
    print("Training Conflict Detector (DeBERTa-v3-base)")
    print("=" * 60)
    
    # Load paths
    base_dir = os.path.dirname(os.path.dirname(__file__))
    train_path = os.path.join(base_dir, "data", "processed", "detector_train.jsonl")
    val_path = os.path.join(base_dir, "data", "processed", "detector_val.jsonl")
    test_path = os.path.join(base_dir, "data", "processed", "detector_test.jsonl")
    
    # Load data
    print("Loading datasets...")
    train_data = load_jsonl(train_path)
    val_data = load_jsonl(val_path)
    test_data = load_jsonl(test_path)
    
    # Initialize tokenizer and model
    print("Initializing model...")
    tokenizer, model = get_detector_model()
    
    train_dataset = ConflictDataset(train_data, tokenizer)
    val_dataset = ConflictDataset(val_data, tokenizer)
    test_dataset = ConflictDataset(test_data, tokenizer)
    
    # Settings as specified in prompt
    batch_size = 16
    epochs = 3
    lr = 1e-5
    eps = 1e-6 # Critical to prevent NaN loss
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.float()  # Force all parameters to FP32 to fix the 'unscale FP16 gradients' error
    print(f"Using device: {device}")
    
    optimizer = AdamW(model.parameters(), lr=lr, eps=eps)
    scaler = GradScaler('cuda')
    
    best_f1 = 0.0
    model_save_path = os.path.join(base_dir, "model", "best_detector")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch in progress_bar:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            with autocast('cuda'):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_train_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        all_preds = []
        all_labels = []
        val_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                with autocast('cuda'):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    val_loss += outputs.loss.item()
                    
                preds = torch.argmax(outputs.logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        val_f1 = f1_score(all_labels, all_preds)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1} Results: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val F1: {val_f1:.4f}")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            os.makedirs(model_save_path, exist_ok=True)
            model.save_pretrained(model_save_path)
            tokenizer.save_pretrained(model_save_path)
            print(f"--> Saved new best model to {model_save_path}")

    # Final Testing
    print("\nEvaluating on Test Set...")
    model = AutoModelForSequenceClassification.from_pretrained(model_save_path)
    model.to(device)
    model.eval()
    
    test_preds = []
    test_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            with autocast('cuda'):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                
            preds = torch.argmax(outputs.logits, dim=-1)
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
            
    test_f1 = f1_score(test_labels, test_preds)
    print("=" * 60)
    print(f"Final Test F1: {test_f1:.4f} (Target: > 0.75)")
    print("=" * 60)

if __name__ == "__main__":
    train_detector()

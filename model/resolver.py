import os
import json
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset

def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def format_dataset(data, tokenizer):
    """Format data into a Dataset for Causal LM training."""
    formatted = []
    for item in data:
        # Instruction format explicitly requested
        instruction = item["instruction"]
        output = item["output"]
        
        # Append EOS token to teach the model when to stop generating
        full_text = f"{instruction} {output}{tokenizer.eos_token}"
        
        tokenized = tokenizer(full_text, truncation=True, max_length=1024, padding=False)
        
        formatted.append({
            "input_ids": tokenized['input_ids'],
            "attention_mask": tokenized['attention_mask']
        })
        
    return Dataset.from_list(formatted)

def train_resolver():
    print("=" * 60)
    print("Training Conflict Resolver (Mistral-7B + QLoRA)")
    print("=" * 60)
    
    # Paths
    base_dir = os.path.dirname(os.path.dirname(__file__))
    train_path = os.path.join(base_dir, "data", "processed", "reader_train.jsonl")
    val_path = os.path.join(base_dir, "data", "processed", "reader_val.jsonl")
    
    print("[1/5] Loading data...")
    train_data = load_jsonl(train_path)
    val_data = load_jsonl(val_path)
    
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    
    print("[2/5] Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Mistral does not have an official pad token, so we use eos_token
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    print("[3/5] Formatting Datasets...")
    train_dataset = format_dataset(train_data, tokenizer)
    val_dataset = format_dataset(val_data, tokenizer)
    
    print("[4/5] Loading Model with 4-bit NF4 Quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    
    # Enable gradient checkpointing to save massive VRAM on T4
    model.gradient_checkpointing_enable()
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # LoRA config as explicitly requested
    print("[5/5] Applying LoRA Adapter...")
    peft_config = LoraConfig(
        lora_alpha=32,
        lora_dropout=0.05,
        r=16,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"]
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Training Arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(base_dir, "model", "resolver_checkpoints"),
        num_train_epochs=2,
        per_device_train_batch_size=1,    # Reduced to 1 for T4 16GB VRAM limit
        gradient_accumulation_steps=16,   # Increased to 16 to keep effective batch size the same (16)
        learning_rate=2e-4,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch", # Fixed deprecated argument for new transformers version
        optim="paged_adamw_8bit",
        fp16=True, 
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="cosine"
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    print("\n--- Starting Training ---")
    trainer.train()
    
    print("\n--- Saving LoRA Adapter ---")
    save_path = os.path.join(base_dir, "model", "best_resolver")
    os.makedirs(save_path, exist_ok=True)
    trainer.model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"[SUCCESS] Resolver adapter saved successfully to {save_path}")

if __name__ == "__main__":
    train_resolver()

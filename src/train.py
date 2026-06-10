import json
import torch
from transformers import (
    T5Tokenizer, T5ForConditionalGeneration,
    TrainingArguments, Trainer, DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import Dataset
import os

class StyleTransferDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=384):
        self.examples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                style = data["style"].lower()
                corrupted = data["corrupted"]
                original = data["original"]
                prompt = f"Restore and rewrite the following text in a {style} style:\n{corrupted}"
                self.examples.append((prompt, original))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        prompt, target = self.examples[idx]
        model_inputs = self.tokenizer(
            prompt,
            truncation=True,
            padding=False,
            max_length=self.max_length,
            return_tensors=None
        )
        # No as_target_tokenizer, just tokenize target normally
        labels = self.tokenizer(
            target,
            truncation=True,
            padding=False,
            max_length=self.max_length,
            return_tensors=None
        )
        return {
            "input_ids": model_inputs["input_ids"],
            "attention_mask": model_inputs["attention_mask"],
            "labels": labels["input_ids"]
        }

def train_incremental():
    base_model_path = "google/flan-t5-large"
    dataset_path = "data/datasets/style_corrupt_pairs.jsonl"
    adapter_path = "models/final/lora_style_rewriter"
    output_dir = "models/checkpoints/lora_style"

    tokenizer = T5Tokenizer.from_pretrained(base_model_path)
    model = T5ForConditionalGeneration.from_pretrained(base_model_path)

    # Apply LoRA config
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=8,
        lora_alpha=16,
        target_modules=["q", "v", "k", "o"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    model.gradient_checkpointing_enable()
    model.print_trainable_parameters()

    dataset = StyleTransferDataset(dataset_path, tokenizer)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_steps=50,
        logging_steps=20,
        save_strategy="epoch",
        eval_strategy="no",
        save_total_limit=2,
        fp16=False,
        dataloader_num_workers=0,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    print(f"Training on {len(dataset)} total pairs.")
    trainer.train()   # no resume_from_checkpoint to avoid errors

    # Save adapter
    os.makedirs(adapter_path, exist_ok=True)
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"LoRA adapter saved to {adapter_path}")

if __name__ == "__main__":
    train_incremental()

import json
import torch
from transformers import (
    T5Tokenizer, T5ForConditionalGeneration,
    TrainingArguments, Trainer, DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import Dataset
import os

class StyleContinuationDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=512):
        self.examples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                style = data["style"].lower()
                prompt_text = data["prompt"]
                target_text = data["target"]
                full_prompt = f"<STYLE={style.upper()}>\nContinue the following text in the same style:\n\n{prompt_text}"
                self.examples.append((full_prompt, target_text))
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

def train_continuation():
    base_model_path ="google/flan-t5-large"
    dataset_path = "data/datasets/style_continuation.jsonl"
    output_dir = "models/checkpoints/lora_continuation"
    final_adapter_dir = "models/final/lora_style_rewriter"

    tokenizer = T5Tokenizer.from_pretrained(base_model_path)
    model = T5ForConditionalGeneration.from_pretrained(base_model_path)

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

    dataset = StyleContinuationDataset(dataset_path, tokenizer, max_length=512)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_steps=100,
        logging_steps=20,
        save_strategy="steps",
        save_steps=250,
        save_total_limit=5,
        eval_strategy="no",
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

    # Resume from latest checkpoint if exists
    resume_checkpoint = None
    if os.path.exists(output_dir):
        checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
        if checkpoints:
            latest = max(checkpoints, key=lambda x: int(x.split("-")[1]))
            resume_checkpoint = os.path.join(output_dir, latest)
            print(f"Resuming from {resume_checkpoint}")

    print(f"Training on {len(dataset)} continuation pairs.")
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    os.makedirs(final_adapter_dir, exist_ok=True)
    model.save_pretrained(final_adapter_dir)
    tokenizer.save_pretrained(final_adapter_dir)
    print(f"Final LoRA adapter saved to {final_adapter_dir}")

if __name__ == "__main__":
    train_continuation()
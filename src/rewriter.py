from transformers import T5Tokenizer, T5ForConditionalGeneration
from peft import PeftModel
import torch
import yaml
import os

class Rewriter:
    def __init__(
        self,
        config_path="configs/model.yaml",
        lora_path="models/checkpoints/lora_style/checkpoint-1261-20260610T103833Z-3-001/checkpoint-1261"
    ):
        print(f"Loading config from: {os.path.abspath(config_path)}")

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        print("Loaded config:", config)

        if not config:
            raise ValueError("model.yaml is empty")

        if "model" not in config:
            raise ValueError(
                f"Expected 'model' key in config. Found keys: {list(config.keys())}"
            )

        base_model_path = config["model"]["local_path"]

        print(f"Loading base model from {base_model_path}")

        self.tokenizer = T5Tokenizer.from_pretrained(base_model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(base_model_path)

        if os.path.exists(lora_path):
            print(f"Loading LoRA adapter from {lora_path}")
            self.model = PeftModel.from_pretrained(
                self.model,
                lora_path
            )
        else:
            print(f"LoRA path not found: {lora_path}")
            print("Using base model only.")

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model.to(self.device)
        self.model.eval()

    def rewrite(self, text, style):
        prompt = (
            f"<STYLE={style.upper()}>\n"
            f"Rewrite the following text while preserving meaning:\n"
            f"{text}"
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        inputs = {
            k: v.to(self.device)
            for k, v in inputs.items()
        }

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
                repetition_penalty=1.1
            )

        return self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        ).strip()
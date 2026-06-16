import json

from datasets import Dataset
from peft import LoraConfig, get_peft_model
from schema import LabeledReport
from transformers import AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer

# Load base model
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

def _to_messages(report, category, severity):
    return [
        {
            "role": "user",
            "content": (
                f"Voici un rapport d'incident industriel :\n\n{report}\n\n"
                f"En te basant uniquement sur le contenu du rapport, sans faire d'inférences externes, réponds aux questions suivantes :\n"
                f"1. Quelle est la catégorie de l'incident ?\n"
                f"2. Quelle est la sévérité de l'incident ?\n\n"
                 "Réponds uniquement en JSON selon le schéma fourni."
            ),
        },
        {
            "role": "assistant",
            "content": json.dumps({"category": category, "severity": severity}, ensure_ascii=False),
        }
    ]

ds = []
with open("data/reports.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        report = LabeledReport.model_validate_json(line.strip())
        ds.append({"messages": _to_messages(report.report, report.category, report.severity)})

dataset = Dataset.from_list(ds)

# Apply PEFT configuration
peft_config = LoraConfig(
    r=32,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)

# Pass PEFT-wrapped model to trainer
sft_config = SFTConfig(
    output_dir="models/qwen2.5-1.5b-lora",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    learning_rate=2e-4,
    logging_steps=10,
    save_strategy="epoch",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=sft_config,
)

trainer.train()
trainer.save_model()


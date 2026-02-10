from IPython.display import clear_output
import csv
import os
from datetime import datetime

clear_output()

from datasets import load_dataset

dataset = load_dataset(path="HuggingFaceTB/smoltalk", name="everyday-conversations")
clear_output()
dataset

dataset["train"][0]

import torch
from trl import clone_chat_template
from transformers import AutoModelForCausalLM, AutoTokenizer

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# Load the model and tokenizer
model_name = "HuggingFaceTB/smolLM2-135M"
model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_name
).to(device)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)

# Define the chat template
source_tokenizer_path = "HuggingFaceTB/SmolLM2-360M-Instruct"
model, tokenizer, added_tokens = clone_chat_template(model=model, tokenizer=tokenizer, source_tokenizer_path=source_tokenizer_path)

# Set the directory where adapter weights will be saved
finetuned_name = "SmolLM2-F2-MyDataset"

clear_output()

from transformers import pipeline

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)

prompts = [
    "What is the capital of Germany? Explain why thats the case and if it was different in the past?",
    "Write a Python function to calculate the factorial of a number.",
    "A rectangular garden has a length of 25 feet and a width of 15 feet. If you want to build a fence around the entire garden, how many feet of fencing will you need?",
    "What is the difference between a fruit and a vegetable? Give examples of each.",
]

def log_to_csv(data, filename='training_metrics.csv'):
    data = {"timestamp": datetime.now().isoformat(), **data}
    file_exists = os.path.isfile(filename)
    with open(filename, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)

def test_inference(prompt, pipe):
    prompt = pipe.tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt = True,
    )
    outputs = pipe(
        prompt,
        max_new_tokens=256,
    )
    return outputs[0]["generated_text"][len(prompt):].strip()

for prompt in prompts:
    print(f"    prompt:\n{prompt}")
    print(f"    response:\n{test_inference(prompt, pipe)}")
    print("." * 50)

""" SFTTrainer provide integration with PEFT wich simplify effective fine-tuning with LoRA """

from peft import LoraConfig

# LoRA configuration
rank_dimension = 6 # Rank of LoRA update matrices (controls compression level)
lora_alpha = (
    8 # Scaling factor for LoRA updates (helps balance between pre-trained and new knowledge)
)
lora_dropout = (
    0.05  # Dropout rate for LoRA layers (reduces overfitting during fine-tuning)
)

peft_config = LoraConfig(
    r = rank_dimension, # LoRA rank (e.g., 4-32)
    lora_alpha=lora_alpha, # Scaling factor for LoRA updates
    lora_dropout=lora_dropout, # Dropout rate for LoRA layers
    target_modules="all-linear",
    task_type="CAUSAL_LM", # task_type: causal language modeling
)

from trl import SFTConfig

args = SFTConfig(
    # Output directory for fine-tuned model
    output_dir=finetuned_name,
    # Training duration
    num_train_epochs=1,
    # Batch size settings
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    # Memory optimization
    gradient_checkpointing=True,
    # Optimizer configuration
    optim="adamw_torch_fused",
    learning_rate=2e-4,
    max_grad_norm=0.3,
    # Learning rate scheduler
    warmup_ratio=0.03,
    lr_scheduler_type="constant",
    # Logging and checkpointing
    logging_steps=10,
    save_strategy="epoch",
    # Precision settings
    bf16=True,
    # Integration settings
    push_to_hub=False,
    report_to="none",
)

from trl import SFTTrainer 

# Create an SFTTrainer with the LoRA configuration
trainer=SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    peft_config=peft_config,
    processing_class=tokenizer,
)

# Resume: load completed checkpoint instead of retraining
checkpoint_path = os.path.join(finetuned_name, "checkpoint-565")
if os.path.isdir(checkpoint_path):
    print(f"Loading completed checkpoint from {checkpoint_path}, skipping training...")
    train_result = trainer.train(resume_from_checkpoint=checkpoint_path)
else:
    print("No checkpoint found, training from scratch...")
    train_result = trainer.train()
log_to_csv(train_result.metrics)
trainer.save_model()

""" Merge the LoRA adapters with the base model to create a single, optimized model """

from peft import LoraConfig, AutoPeftModelForCausalLM

# Loading adapter weights on CPU
lora_model = AutoPeftModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=args.output_dir,
    dtype=torch.float16,
    low_cpu_mem_usage=True,
)

# Merge LoRA with the base model and saving it
merged_model = lora_model.merge_and_unload()
merged_model.save_pretrained(
    args.output_dir, safe_serialization=True, max_shard_size="2GB"
)

merged_pipe = pipeline(
    "text-generation", model=merged_model, tokenizer=tokenizer, device=device
)

for prompt in prompts:
    print(f"    prompt:\n{prompt}")
    print(f"    response:\n{test_inference(prompt, merged_pipe)}")
    print("=" * 50)

""" Analys of change for a specific layer """

model_name = "HuggingFaceTB/SmolLM2-135M"
model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_name,
).to(device)
model

merged_model

# Let's take the same weight matrix and see that it is different.
merged_model.model.layers[0].self_attn.q_proj

merged = merged_model.model.layers[0].self_attn.q_proj.weight.data
merged

# Let's write the low-rank matrices A and B, which were trained using LoRA, into separate variables.
trainer.model.base_model.model.model.layers[0].self_attn.q_proj.lora_A.default

a = trainer.model.base_model.model.model.layers[
    0
].self_attn.q_proj.lora_A.default.weight.data
a

trainer.model.base_model.model.model.layers[0].self_attn.q_proj.lora_B.default

base = model.model.layers[0].self_attn.q_proj.weight.data.to("cpu")
b = trainer.model.base_model.model.model.layers[
    0
].self_attn.q_proj.lora_B.default.weight.data.to("cpu")
a = a.to("cpu")

base + lora_alpha / rank_dimension * (b @ a)

merged


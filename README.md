# Fine-tune NLP Models

Fine-tuning pipeline for small language models using LoRA/QLoRA with Hugging Face ecosystem.

## Scripts

| Script | Description |
|--------|-------------|
| `legal-nlp-finetune.py` | SFT fine-tuning SmolLM2-135M with LoRA on conversational data |
| `finetune_qlora_4b.py` | QLoRA fine-tuning for larger models with GGUF validation |
| `preprocess_sft_pipeline.py` | Data preprocessing pipeline: raw data to train/val/test splits |

## Quick Start

```bash
pip install torch transformers trl peft datasets
python legal-nlp-finetune.py
```

## Pipeline (legal-nlp-finetune.py)

1. Load base model (`HuggingFaceTB/SmolLM2-135M`) and clone chat template
2. Run pre-training inference baseline
3. Fine-tune with LoRA via SFTTrainer
4. Log training metrics to CSV
5. Merge LoRA adapter into base model
6. Run post-training inference comparison
7. Analyze weight changes per layer (LoRA decomposition: `base + alpha/r * B@A`)

### LoRA Config

| Parameter | Value |
|-----------|-------|
| Rank (r) | 6 |
| Alpha | 8 |
| Dropout | 0.05 |
| Target | all-linear |

### Training Config

| Parameter | Value |
|-----------|-------|
| Dataset | `HuggingFaceTB/smoltalk` (everyday-conversations) |
| Epochs | 1 |
| Batch size | 2 (x2 gradient accumulation) |
| Optimizer | AdamW fused |
| Learning rate | 2e-4 |
| LR scheduler | constant |
| Precision | bf16 |

## Example Training Results (SmolLM2-135M)

```
Steps:              565
Epoch:              1.0
Runtime:            5741s (~1h 36m)
Samples/sec:        0.394
Final train loss:   1.267
Mean token accuracy: 76.3%
```

### Loss Curve

```
Step   Loss    Accuracy
  10   2.926   49.0%
  50   1.581   61.5%
 100   1.292   69.1%
 200   1.187   70.3%
 300   1.133   70.8%
 400   1.100   71.2%
 500   1.148   71.3%
 565   1.067   74.7%  (final)
```

## Output

Training produces `SmolLM2-F2-MyDataset/` (excluded from git):

```
SmolLM2-F2-MyDataset/
├── model.safetensors           # merged model (base + LoRA)
├── adapter_model.safetensors   # LoRA weights only
├── adapter_config.json
├── config.json
├── tokenizer.json
└── checkpoint-565/             # full trainer state for resume
```

## Description model & dataset

Датасет: HuggingFaceTB/smoltalk
Это высококачественный набор данных для дообучения (SFT — Supervised Fine-Tuning), созданный специально для улучшения навыков ведения диалога и следования инструкциям у маленьких моделей.

Состав: Смесь публичных и синтетических данных, включая Smol-Magpie-Ultra (400 тыс. примеров, сгенерированных Llama-3.1-405B), данные для редактирования текста, суммаризации и логических задач.

Особенности:

Многоходовость: Около 70% данных — это длинные диалоги (multi-turn), а не просто одиночные вопросы.
​

Разнообразие: Включает задачи на следование строгим ограничениям (например, «ответь ровно в 3 предложениях»).
​

Мини-версия: Существует подмножество smol-smoltalk, отобранное специально для самых крошечных моделей вроде 135M.

Модель: SmolLM2-135M
Это одна из самых маленьких современных языковых моделей (всего 135 миллионов параметров), входящая в семейство SmolLM2.

Размер и память: Весит всего около 720 МБ в формате bfloat16. Она потребляет экстремально мало ресурсов, что позволяет запускать её полностью офлайн на телефонах или в браузере.

Обучение: Обучена на колоссальном объеме данных — 2 триллиона токенов, включая образовательные материалы (FineWeb-Edu), код (The Stack) и математические тексты.

Производительность: Для своего размера она показывает феноменальные результаты в следовании инструкциям и базовых рассуждениях, превосходя многие модели, которые в 10–20 раз больше её по размеру

## Requirements

- Python 3.10+
- PyTorch (CUDA, MPS, or CPU)
- transformers, trl, peft, datasets

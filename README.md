# Gemma-2B Fine-Tuning with LoRA

A Google Colab notebook for fine-tuning the Gemma-2B language model on quotes using Low-Rank Adaptation (LoRA) and quantization.

## Overview

This notebook demonstrates how to efficiently fine-tune Google's Gemma-2B model on a quotes dataset using:
- **4-bit Quantization** (BitsAndBytes) to reduce memory usage
- **LoRA** (Low-Rank Adaptation) for parameter-efficient training
- **Supervised Fine-Tuning (SFT)** via the TRL library

## Requirements

- Google Colab (GPU access recommended)
- HuggingFace token (stored as `HF_TOKEN` in Colab secrets)
- Python 3.12+

## Key Dependencies

```
transformers
trl
peft
accelerate
bitsandbytes
datasets
torch
```

## Workflow

1. **Setup**: Install dependencies and authenticate with HuggingFace
2. **Quantization**: Configure 4-bit quantization for the Gemma-2B model
3. **Model Loading**: Load the tokenizer and model with quantization
4. **Data Preparation**: Load the English Quotes dataset and format training examples
5. **LoRA Configuration**: Set up LoRA adapters for specific model layers
6. **Fine-Tuning**: Train the model using SFTTrainer with gradient accumulation
7. **Inference**: Generate text predictions using the fine-tuned model

## Configuration

**LoRA Settings:**
- Rank (r): 8
- Target modules: q_proj, o_proj, k_proj, v_proj, gate_proj, up_proj, down_proj

**Training Settings:**
- Batch size: 1 (per device)
- Gradient accumulation steps: 4
- Learning rate: 2e-4
- Max steps: 100
- Optimizer: paged_adamw_8bit

## Usage

Run cells sequentially in Google Colab. Ensure the HF_TOKEN secret is set before executing the authentication cell.

## Output

The fine-tuned model weights are saved to the `outputs/` directory. The model can then generate text continuations for custom prompts (e.g., "Quote: A woman is like a tea bag...").

## Notes

- This setup uses 4-bit quantization to fit the model in Colab's GPU memory
- LoRA reduces trainable parameters while maintaining performance
- The model is fine-tuned on famous quotes to demonstrate text generation capabilities

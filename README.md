# Source Code Documentation

This directory contains the core implementation of the Custom Language Model training pipeline.

## 📁 Directory Structure

### 1. `fine_tune.py`
Main fine-tuning script that orchestrates the training process.
- `prepare_model_and_tokenizer()`: Initializes model and tokenizer
- `load_and_process_dataset()`: Handles dataset preparation
- `train_model()`: Manages the training process
- `fine_tune()`: Main entry point for fine-tuning

### 2. `main.py`
Entry point of the application that coordinates:
- Model fine-tuning
- Model saving
- Hugging Face Hub upload

### 3. `/huggingface`
Utilities for Hugging Face integration:
- `upload.py`: Handles model upload to Hugging Face Hub

### 4. `/model`
Model-related implementations:
- `build_model.py`: Custom transformer model architecture
- `save_model.py`: Model and tokenizer saving utilities

### 5. `/training`
Training-related utilities:
- `dataset_preparation.py`: Dataset loading and processing
- `trainer.py`: Training configuration and execution

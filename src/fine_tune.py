from datasets import load_dataset
from transformers import Trainer, TrainingArguments
from model.build_model import CustomLLM, get_gpt2_tokenizer, create_model_config
from training.dataset_preparation import prepare_dataset, tokenize_dataset

def prepare_model_and_tokenizer():
    # Get GPT-2 tokenizer
    tokenizer = get_gpt2_tokenizer()
    
    # Create model configuration
    config = create_model_config()
    
    # Initialize the model
    model = CustomLLM(config)
    
    return model, tokenizer

def load_and_process_dataset(tokenizer):
    # Load wikitext dataset
    dataset = prepare_dataset("wikitext-2-raw-v1")
    
    # Tokenize the dataset
    tokenized_dataset = tokenize_dataset(dataset, tokenizer)
    
    return tokenized_dataset

def train_model(model, tokenized_dataset, output_dir="./results"):
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        save_steps=50,
        logging_steps=50,
        save_total_limit=2,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
    )
    
    trainer.train()
    return trainer

def fine_tune():
    # Initialize model and tokenizer
    model, tokenizer = prepare_model_and_tokenizer()
    
    # Load and process dataset
    tokenized_dataset = load_and_process_dataset(tokenizer)
    
    # Train the model
    trainer = train_model(model, tokenized_dataset)
    
    return model, tokenizer, trainer

if __name__ == "__main__":
    fine_tune() 
# Done

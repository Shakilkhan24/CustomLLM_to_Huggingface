from transformers import Trainer, TrainingArguments

def get_training_args(output_dir, epochs=3):
    """
    Create training arguments with default values
    """
    return TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=epochs,
        weight_decay=0.01,
    )

def fine_tune_model(model, training_args, train_dataset, eval_dataset):
    """
    Fine-tune the model using the provided datasets
    """
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    return trainer.train() 
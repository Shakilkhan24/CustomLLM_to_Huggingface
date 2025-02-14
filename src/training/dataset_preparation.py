from datasets import load_dataset

def prepare_dataset(dataset_name):
    """
    Load and prepare the dataset for training
    """
    return load_dataset(dataset_name)

def tokenize_dataset(dataset, tokenizer):
    """
    Tokenize the dataset using the provided tokenizer
    """
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)
    
    return dataset.map(tokenize_function, batched=True) 
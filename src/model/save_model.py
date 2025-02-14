from transformers import AutoModelForCausalLM, AutoTokenizer

def save_model_and_tokenizer(model, tokenizer, directory):
    """
    Save the model and tokenizer to a specified directory
    """
    model.save_pretrained(directory)
    tokenizer.save_pretrained(directory)

def load_base_model(model_path, tokenizer_path):
    """
    Load the base model and tokenizer from specified paths
    """
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    return model, tokenizer 
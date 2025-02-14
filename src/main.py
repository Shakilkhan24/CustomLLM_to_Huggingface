from fine_tune import fine_tune
from model.save_model import save_model_and_tokenizer
from huggingface.upload import upload_to_huggingface

def main():
    # Fine-tune the model
    model, tokenizer, trainer = fine_tune()
    
    # Save and upload fine-tuned model
    save_model_and_tokenizer(model, tokenizer, "fine_tuned_model_directory")
    upload_to_huggingface(
        "fine_tuned_model_directory",
        "Shakil2448868",
        "testing-llm-01"
    )

if __name__ == "__main__":
    main() 
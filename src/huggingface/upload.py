from huggingface_hub import HfApi, create_repo

def upload_to_huggingface(folder_path, username, model_name):
    """
    Upload a model to Hugging Face
    """
    repo_id = f"{username}/{model_name}"
    create_repo(repo_id, exist_ok=True)
    
    api = HfApi()
    api.upload_folder(
        folder_path=folder_path,
        repo_id=repo_id,
        repo_type="model"
    )
    return repo_id 
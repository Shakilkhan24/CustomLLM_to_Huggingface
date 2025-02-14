import torch
from torch import nn
from transformers import PreTrainedTokenizerFast, PreTrainedModel, GPT2Tokenizer
from typing import Optional, List

class CustomLLM(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        self.num_heads = config.num_attention_heads
        
        # Embedding layer
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.hidden_size,
                nhead=self.num_heads,
                dim_feedforward=4 * self.hidden_size,
                batch_first=True
            ) for _ in range(self.num_layers)
        ])
        
        # Output layer
        self.output = nn.Linear(self.hidden_size, self.vocab_size)
        
    def forward(self, input_ids, attention_mask=None):
        # Embed the input
        x = self.embedding(input_ids)
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x)
            
        # Get logits
        logits = self.output(x)
        
        return {"logits": logits}

def get_gpt2_tokenizer():
    """
    Get the pre-trained GPT-2 tokenizer
    """
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def create_model_config():
    """
    Create a configuration for the model using GPT-2 vocabulary size
    """
    from transformers import PretrainedConfig
    
    class CustomConfig(PretrainedConfig):
        model_type = "custom_llm"
        
        def __init__(
            self,
            vocab_size=50257,  # GPT-2 vocabulary size
            hidden_size=768,
            num_layers=6,
            num_attention_heads=12,
            **kwargs
        ):
            super().__init__(**kwargs)
            self.vocab_size = vocab_size
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.num_attention_heads = num_attention_heads
    
    return CustomConfig() 
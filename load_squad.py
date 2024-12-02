import os
from datasets import load_dataset
from transformers import AutoTokenizer
from dataloader import get_tokenizer  # Import tokenizer helper

def preprocess_squad(example, tokenizer, block_size=1024):
    """
    Preprocess SQuAD examples into question-context pairs with tokenized input.
    """
    question = example['question']
    context = example['context']
    input_text = f"Question: {question} Context: {context}"
    
    # Tokenize input text
    tokens = tokenizer(
        input_text,
        max_length=block_size,
        padding="max_length",
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_token_type_ids=True,
    )
    
    # Add labels (answers) for fine-tuning or evaluation
    if 'answers' in example and example['answers']['text']:
        tokens['labels'] = tokenizer(
            example['answers']['text'][0],  # Take the first answer as the label
            max_length=block_size,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt"
        )['input_ids']
    
    return tokens

def load_squad_data(config, mode='train'):
    """
    Load and preprocess the SQuAD dataset for MDLM.
    """
    # Load the tokenizer
    tokenizer = get_tokenizer(config)
    
    # Load the SQuAD dataset
    squad_dataset = load_dataset("squad", cache_dir=config.data.cache_dir)
    squad_data = squad_dataset[mode]
    
    # Preprocess and tokenize the dataset
    tokenized_dataset = squad_data.map(
        lambda example: preprocess_squad(example, tokenizer, block_size=config.model.length),
        batched=True,
        remove_columns=['id', 'title', 'answers'],  # Remove unused columns
        desc=f"Tokenizing SQuAD {mode} data",
    )
    
    # Convert to PyTorch format and return
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return tokenized_dataset

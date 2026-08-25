"""
SFT Expand Dataset for GSM8K and similar structured datasets
Supports loading from parquet files with flexible prompt/response key names
"""

import os
import torch
from torch.utils.data import Dataset
from datasets import load_dataset, load_from_disk, concatenate_datasets
from typing import List, Optional, Any, Union
import glob
from pathlib import Path


class SFTExpandDataset(Dataset):
    """
    A dataset class for Supervised Fine-Tuning (SFT) that supports:
    - Loading from parquet files
    - Flexible prompt/response key mapping
    - GSM8K and other math reasoning datasets
    - Proper tokenization with attention masks and loss masks
    """
    
    def __init__(
        self,
        parquet_files: Union[List[str], str],
        tokenizer: Any,
        prompt_key: str = "question",
        response_key: str = "answer",
        max_length: int = 2048,
        truncation: bool = True,
        system_prompt: Optional[str] = None,
    ):
        """
        Initialize the SFT Expand Dataset.
        
        Args:
            parquet_files (List[str]): List of parquet file paths to load
            tokenizer (Any): HuggingFace tokenizer instance
            prompt_key (str): Key name for the prompt/question in the dataset
            response_key (str): Key name for the response/answer in the dataset
            max_length (int): Maximum sequence length for tokenization
            truncation (bool): Whether to truncate sequences exceeding max_length
            system_prompt (Optional[str]): Optional system prompt to prepend to each example
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.prompt_key = prompt_key
        self.response_key = response_key
        self.max_length = max_length
        self.truncation = truncation
        self.system_prompt = system_prompt or "You are a helpful assistant that solves math problems step by step."
        
        # Normalize parquet_files to a list
        if isinstance(parquet_files, str):
            parquet_files = [parquet_files]
        # Load dataset from parquet files
        self.data = self._load_data(parquet_files)
        
    def _expand_input_paths(self, file_path: str) -> List[str]:
        """Expand a single input path into concrete file paths.
        Supports wildcard patterns and directories containing parquet files.
        """
        file_path = str(file_path)
        paths: List[str] = []
        # Wildcard patterns
        if any(ch in file_path for ch in ["*", "?"]):
            paths = [p for p in glob.glob(file_path) if os.path.exists(p)]
        # Directory: collect parquet files or keep directory for load_from_disk
        elif os.path.isdir(file_path):
            parquet_list = sorted(Path(file_path).glob('*.parquet'))
            if parquet_list:
                paths = [str(p) for p in parquet_list]
            else:
                paths = [file_path]
        else:
            paths = [file_path]
        return paths

    def _load_one(self, candidate: str):
        """Load a single dataset candidate path."""
        if candidate.endswith('.parquet'):
            return load_dataset('parquet', data_files=candidate, split='train')
        elif os.path.isdir(candidate):
            dataset = load_from_disk(candidate)
            if hasattr(dataset, 'keys'):
                dataset = dataset['train'] if 'train' in dataset else list(dataset.values())[0]
            return dataset
        elif candidate.endswith('.jsonl') or candidate.endswith('.json'):
            return load_dataset('json', data_files=candidate, split='train')
        else:
            print(f"Warning: Unsupported file format: {candidate}")
            return None

    def _load_data(self, parquet_files: List[str]) -> List[dict]:
        """Load data from parquet/json files or dataset directories."""
        all_datasets = []
        for file_path in parquet_files:
            expanded = self._expand_input_paths(file_path)
            if not expanded:
                print(f"Warning: No files matched: {file_path}")
                continue
            for candidate in expanded:
                if not os.path.exists(candidate):
                    print(f"Warning: File not found: {candidate}")
                    continue
                try:
                    dataset = self._load_one(candidate)
                    if dataset is None:
                        continue
                    all_datasets.append(dataset)
                    print(f"Loaded {len(dataset)} examples from {candidate}")
                except Exception as e:
                    print(f"Error loading {candidate}: {e}")
                    continue

        if not all_datasets:
            raise ValueError(f"No valid data files found in: {parquet_files}")

        combined_dataset = (
            concatenate_datasets(all_datasets) if len(all_datasets) > 1 else all_datasets[0]
        )
        print(f"Total dataset size: {len(combined_dataset)} examples")
        return list(combined_dataset)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Get a single example and tokenize it.
        
        Returns:
            dict: Contains input_ids, attention_mask, labels, position_ids, and loss_mask
        """
        example = self.data[idx]
        
        # Extract prompt and response using the specified keys
        prompt = example.get(self.prompt_key, "")
        response = example.get(self.response_key, "")
        
        # Construct the full conversation
        # Format: System + User message + Assistant response
        full_text = f"{self.system_prompt}\n\nQuestion: {prompt}\n\nAnswer: {response}"
        
        # Tokenize the full text
        encoded_full = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=self.truncation,
            padding=False,  # We'll pad in collate_fn
            return_tensors=None,  # Return lists, not tensors
        )
        
        # Tokenize just the prompt to calculate the prompt length
        prompt_text = f"{self.system_prompt}\n\nQuestion: {prompt}\n\nAnswer: "
        encoded_prompt = self.tokenizer(
            prompt_text,
            max_length=self.max_length,
            truncation=self.truncation,
            padding=False,
            return_tensors=None,
        )
        
        prompt_length = len(encoded_prompt['input_ids'])
        total_length = len(encoded_full['input_ids'])
        
        # Create input_ids and labels
        input_ids = encoded_full['input_ids']
        attention_mask = encoded_full['attention_mask']
        
        # Labels: same as input_ids for causal LM training
        labels = input_ids.copy()
        
        # Create loss_mask: only compute loss on the response tokens (after prompt)
        # loss_mask[i] = 1 if we want to compute loss on token i, 0 otherwise
        loss_mask = [0] * prompt_length + [1] * (total_length - prompt_length)
        
        # Create position_ids (standard incremental position)
        position_ids = list(range(total_length))
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'position_ids': position_ids,
            'loss_mask': loss_mask,
        }
    
    @staticmethod
    def collate_fn(batch):
        """
        Collate function to pad batches dynamically.
        
        Args:
            batch: List of dicts from __getitem__
            
        Returns:
            dict: Batched and padded tensors
        """
        # Find max length in this batch
        max_len = max(len(item['input_ids']) for item in batch)
        
        # Pad all sequences to max_len
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        batch_position_ids = []
        batch_loss_mask = []
        
        for item in batch:
            seq_len = len(item['input_ids'])
            padding_len = max_len - seq_len
            
            # Pad input_ids (pad with tokenizer.pad_token_id or 0)
            pad_token_id = 0  # You may want to use tokenizer.pad_token_id
            batch_input_ids.append(
                item['input_ids'] + [pad_token_id] * padding_len
            )
            
            # Pad attention_mask (0 for padding)
            batch_attention_mask.append(
                item['attention_mask'] + [0] * padding_len
            )
            
            # Pad labels (usually -100 for ignored positions)
            batch_labels.append(
                item['labels'] + [-100] * padding_len
            )
            
            # Pad position_ids
            batch_position_ids.append(
                item['position_ids'] + [0] * padding_len
            )
            
            # Pad loss_mask (0 for padding)
            batch_loss_mask.append(
                item['loss_mask'] + [0] * padding_len
            )
        
        return {
            'input_ids': torch.tensor(batch_input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(batch_attention_mask, dtype=torch.long),
            'labels': torch.tensor(batch_labels, dtype=torch.long),
            'position_ids': torch.tensor(batch_position_ids, dtype=torch.long),
            'loss_mask': torch.tensor(batch_loss_mask, dtype=torch.long),
        }


class GSM8KDataset(SFTExpandDataset):
    """
    Specialized dataset class for GSM8K (Grade School Math 8K).
    
    GSM8K format:
    - question: The math problem
    - answer: Step-by-step solution with final answer
    
    This class is a convenience wrapper around SFTExpandDataset with
    GSM8K-specific defaults.
    """
    
    def __init__(
        self,
        parquet_files: List[str],
        tokenizer: Any,
        max_length: int = 2048,
        truncation: bool = True,
    ):
        """
        Initialize GSM8K dataset.
        
        Args:
            parquet_files (List[str]): List of parquet file paths
            tokenizer: HuggingFace tokenizer
            max_length (int): Maximum sequence length
            truncation (bool): Whether to truncate long sequences
        """
        system_prompt = (
            "You are a math expert. You will be given a question to solve. "
            "Solve it step by step. Wrap the final answer in \\boxed{}."
        )
        
        super().__init__(
            parquet_files=parquet_files,
            tokenizer=tokenizer,
            prompt_key="question",
            response_key="answer",
            max_length=max_length,
            truncation=truncation,
            system_prompt=system_prompt,
        )


def test_dataset():
    """Simple test function to verify the dataset works correctly."""
    from transformers import AutoTokenizer
    
    # Load a tokenizer (adjust path as needed)
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        trust_remote_code=True
    )
    
    # Create a dummy parquet file for testing
    import pandas as pd
    test_data = pd.DataFrame([
        {
            "question": "What is 2 + 2?",
            "answer": "Let me solve this step by step.\nStep 1: We need to add 2 and 2.\nStep 2: 2 + 2 = 4\nThe answer is \\boxed{4}."
        },
        {
            "question": "If a train travels 60 miles per hour for 2 hours, how far does it travel?",
            "answer": "Let me solve this.\nStep 1: Speed = 60 mph\nStep 2: Time = 2 hours\nStep 3: Distance = Speed × Time = 60 × 2 = 120 miles\nThe answer is \\boxed{120}."
        }
    ])
    
    test_file = "/tmp/test_gsm8k.parquet"
    test_data.to_parquet(test_file)
    
    # Test SFTExpandDataset
    dataset = SFTExpandDataset(
        parquet_files=[test_file],
        tokenizer=tokenizer,
        prompt_key="question",
        response_key="answer",
        max_length=512,
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test __getitem__
    item = dataset[0]
    print(f"\nFirst item keys: {item.keys()}")
    print(f"Input IDs shape: {len(item['input_ids'])}")
    print(f"Prompt tokens (loss_mask=0): {sum(1 for x in item['loss_mask'] if x == 0)}")
    print(f"Response tokens (loss_mask=1): {sum(1 for x in item['loss_mask'] if x == 1)}")
    
    # Test collate_fn
    batch = [dataset[i] for i in range(2)]
    collated = SFTExpandDataset.collate_fn(batch)
    print(f"\nCollated batch keys: {collated.keys()}")
    print(f"Batch input_ids shape: {collated['input_ids'].shape}")
    print(f"Batch attention_mask shape: {collated['attention_mask'].shape}")
    
    # Cleanup
    os.remove(test_file)
    print("\nTest passed!")


if __name__ == "__main__":
    test_dataset()

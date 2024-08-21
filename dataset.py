import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, tokenized_data, max_seq_len, pad_id):
        self.data = tokenized_data
        self.max_seq_len = max_seq_len
        self.pad_id = pad_id

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        seq_idx = 0
        while idx >= len(self.data[seq_idx]) - 1:
            idx -= (len(self.data[seq_idx]) - 1)
            seq_idx += 1
        
        tokens = self.data[seq_idx]
        
        input_token = torch.tensor(tokens[idx], dtype=torch.long)
        target_token = torch.tensor(tokens[idx + 1], dtype=torch.long)


        return input_token, target_token
import torch
from torch.utils.data import DataLoader
from dataset import TextDataset
from model import llamaModel, ModelArgs
import pickle

import torch
from torch.utils.data import DataLoader
from dataset import TextDataset
from model import llamaModel, ModelArgs
import pickle

with open('split_data.pkl', 'rb') as f:
    _, _, test_data, pad_id = pickle.load(f)

max_seq_len = 128
batch_size = 4
num_epochs = 10 
vocab_size = 100
learning_rate = 0.001
embedding_dim = 256 

test_dataset = TextDataset(test_data, max_seq_len, pad_id)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model_args = ModelArgs(
    dim=128, 
    n_layers=2, 
    n_heads=4,  
    vocab_size=vocab_size,
    max_seq_len=max_seq_len
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = llamaModel(model_args).to(device)
model.load_state_dict(torch.load('best_llama_model.pth', weights_only=True))

model.eval()
test_loss = 0
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=pad_id)
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        inputs = inputs.unsqueeze(1) 
        targets = targets.unsqueeze(1)
        outputs = model(inputs, start_pos=0)
        loss = loss_fn(outputs.view(-1, vocab_size), targets.view(-1))
        test_loss += loss.item()

print(f'Test Loss: {test_loss/len(test_loader):.4f}')

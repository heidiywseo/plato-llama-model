import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import TextDataset
import pickle

from model import ModelArgs, llamaModel

torch.autograd.set_detect_anomaly(True)

with open('split_data.pkl', 'rb') as f:
    train_data, valid_data, test_data, pad_id = pickle.load(f)

max_seq_len = 128
batch_size = 4
num_epochs = 10 
vocab_size = 100
learning_rate = 0.001
embedding_dim = 256 

model_args = ModelArgs(
    dim=128,  # Reduced embedding dimension
    n_layers=2,  # Reduced number of layers
    n_heads=4,  # Reduced number of attention heads
    vocab_size=vocab_size,
    max_seq_len=max_seq_len
)

train_dataset = TextDataset(train_data, max_seq_len, pad_id)
valid_dataset = TextDataset(valid_data, max_seq_len, pad_id)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = llamaModel(model_args).to(device)
loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

best_val_loss = float('inf')
for epoch in range(num_epochs):
    model.train()
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        inputs = inputs.unsqueeze(1)  # Reshape to (batch_size, 1)
        targets = targets.unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(inputs, start_pos=0)
        loss = loss_fn(outputs.view(-1, vocab_size), targets.view(-1))
        loss.backward()
        optimizer.step()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, targets in valid_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.unsqueeze(1) 
            targets = targets.unsqueeze(1)
            outputs = model(inputs, start_pos=0)
            loss = loss_fn(outputs.view(-1, vocab_size), targets.view(-1))
            val_loss = val_loss + loss.item()

    val_loss = val_loss/len(valid_loader)
    print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}')

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_llama_model.pth')


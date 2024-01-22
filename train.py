from dataset import MyDataset, MLMCollator
from model import Config, MaskedLanguageModel
import torch
from torch.utils.data import DataLoader
from torch import nn

vocab_size = 36000
batch_size = 256
seq_length = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path = r"C:\Users\ontheit\Desktop\2a\kakaotech\wikitext-2-raw"
train_dataset = MyDataset(path + "/wiki.train.raw", "tokenizer.json", seq_length)
collate_fn = MLMCollator(train_dataset.tokenizer)
train_dataloader = DataLoader(train_dataset, batch_size=64, collate_fn=collate_fn)

config = Config(
    vocab_size=vocab_size,
    hidden_size=256,
    num_hidden_layers=4,
    num_attention_heads=4,
    intermediate_size=1024,
    max_position_embeddings=seq_length,
)
model = MaskedLanguageModel(config).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
criterion = nn.CrossEntropyLoss()

def train(model: nn.Module, epoch):
    model.train()
    
    total_loss = 0.0
    total_iter = 0
    
    for batch in train_dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        padding_mask = batch["padding_mask"].to(device)
        
        logits = model(input_ids, padding_mask=padding_mask)
        loss = criterion(logits.view(-1, vocab_size), labels.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        total_loss += loss.item()
        total_iter += 1
        
    mean_loss = total_loss / total_iter
    print(f"epoch {epoch+1} : loss {mean_loss:1.4f}")
    
for epoch in range(10):
    train(model, epoch)
import torch
from torch.utils.data import DataLoader
from clmnet_model import CLMNet
from dataset import CRLMDataset

def train():
    model = CLMNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.BCELoss()

    dataset = CRLMDataset('train_data/')
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    model.train()
    for epoch in range(10):
        for images, labels in loader:
            preds = model(images)
            loss = criterion(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

if __name__ == "__main__":
    train()

import torch
import torch.nn as nn
from models.s4.s4d import S4D
from torch.utils.data import DataLoader

from src.dataloaders.datasets.line import LineDataset


class ForecastModel(nn.Module):
    def __init__(self, d_model=64, n_layers=2, dropout=0.0):
        super().__init__()
        self.encoder = nn.Linear(1, d_model)
        self.s4_layers = nn.ModuleList(
            [S4D(d_model, dropout=dropout, transposed=True) for _ in range(n_layers)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.decoder = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.encoder(x)
        x = x.transpose(-1, -2)
        for layer, norm in zip(self.s4_layers, self.norms):
            z, _ = layer(x)
            x = norm((x + z).transpose(-1, -2)).transpose(-1, -2)
        x = x.transpose(-1, -2)
        x_last = x[:, -1]
        out = self.decoder(x_last)
        return out


def train_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Use a 1-step forecast horizon so the model output matches the target
    train_dataset = LineDataset(pred_len=1)
    val_dataset = LineDataset(pred_len=1, seed=1)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    model = ForecastModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for epoch in range(10):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device).squeeze(1)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        avg_train = train_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device).squeeze(1)
                out = model(x)
                loss = criterion(out, y)
                val_loss += loss.item() * x.size(0)
        avg_val = val_loss / len(val_loader.dataset)

        print(f"Epoch {epoch+1}: train {avg_train:.6f}, val {avg_val:.6f}")


if __name__ == "__main__":
    train_model()

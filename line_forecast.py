import torch
import torch.nn as nn
from models.s4.s4d import S4D
from torch.utils.data import DataLoader, Dataset


class LineDataset(Dataset):
    """Synthetic dataset of linear sequences for forecasting."""

    def __init__(self, seq_len=10, pred_len=1, size=1000):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        slope = torch.rand(1) * 2 - 1  # [-1, 1]
        intercept = torch.rand(1) * 2 - 1
        t = torch.arange(self.seq_len + self.pred_len, dtype=torch.float)
        y = slope * t + intercept
        x = y[: self.seq_len].unsqueeze(-1)
        target = y[self.seq_len :].unsqueeze(-1)
        return x, target


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
    dataset = LineDataset()
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = ForecastModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for epoch in range(10):
        model.train()
        total_loss = 0.0
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y.squeeze(1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        avg_loss = total_loss / len(loader.dataset)
        print(f"Epoch {epoch+1}: loss {avg_loss:.6f}")


if __name__ == "__main__":
    train_model()

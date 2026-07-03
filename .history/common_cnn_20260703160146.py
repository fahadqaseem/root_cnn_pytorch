import os
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class EventDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return torch.tensor(self.X[i], dtype=torch.float32), torch.tensor(self.y[i], dtype=torch.float32)


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3,16,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(32*8*8,64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64,1)
        )

    def forward(self,x):
        return self.net(x)


def read_meta():
    meta = {}
    with open("meta.txt") as f:
        for line in f:
            k,v = line.strip().split("=")
            meta[k] = int(v) if v.isdigit() else float(v)
    return meta


def run_training(target_name, target_index):

    meta = read_meta()
    N = meta["n_events"]
    ETA = meta["eta_bins"]
    PHI = meta["phi_bins"]

    X = np.fromfile("images.bin", dtype=np.float32).reshape(N,3,ETA,PHI)
    Y = np.fromfile("targets.bin", dtype=np.float32).reshape(N, 5)

    y = Y[:,target_index].reshape(-1,1)

    # remove bad
    mask = np.isfinite(X).reshape(N,-1).all(axis=1) & np.isfinite(Y).all(axis=1)
    X = X[mask]
    Y = Y[mask]
    y = y[mask]

    N = len(y)
    idx = np.arange(N)

    # scale
    x_scaler = StandardScaler()
    Xs = x_scaler.fit_transform(X.reshape(N,-1)).reshape(N,3,ETA,PHI)

    y_scaler = StandardScaler()
    ys = y_scaler.fit_transform(y)

    # split
    i_tr, i_te = train_test_split(idx, test_size=0.2, random_state=42)
    i_tr, i_val = train_test_split(i_tr, test_size=0.2, random_state=42)

    train_loader = DataLoader(EventDataset(Xs[i_tr], ys[i_tr]), batch_size=256, shuffle=True)
    val_loader   = DataLoader(EventDataset(Xs[i_val], ys[i_val]), batch_size=256)
    test_loader  = DataLoader(EventDataset(Xs[i_te], ys[i_te]), batch_size=256)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SimpleCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    best = 1e9
    patience = 10
    bad = 0

    for ep in range(50):
        model.train()
        tl = 0

        for xb,yb in train_loader:
            xb,yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred,yb)

            opt.zero_grad()
            loss.backward()
            opt.step()
            tl += loss.item()

        model.eval()
        vl = 0
        with torch.no_grad():
            for xb,yb in val_loader:
                xb,yb = xb.to(device), yb.to(device)
                vl += loss_fn(model(xb),yb).item()

        print(f"{target_name} Epoch {ep+1} train {tl:.4f} val {vl:.4f}")

        if vl < best:
            best = vl
            bad = 0
            torch.save(model.state_dict(), f"best_{target_name}.pt")
        else:
            bad += 1

        if bad >= patience:
            break

    # evaluate
    model.load_state_dict(torch.load(f"best_{target_name}.pt", map_location=device))
    model.eval()

    preds, trues, ids = [], [], []

    with torch.no_grad():
        for i,(xb,yb) in enumerate(test_loader):
            xb = xb.to(device)
            pred = model(xb).cpu().numpy()

            preds.append(pred)
            trues.append(yb.numpy())
            ids.extend(i_te[i*256:i*256+len(xb)])

    preds = y_scaler.inverse_transform(np.vstack(preds)).flatten()
    trues = y_scaler.inverse_transform(np.vstack(trues)).flatten()

    rmse = np.sqrt(mean_squared_error(trues,preds))
    r2   = r2_score(trues,preds)

    print(target_name, "RMSE", rmse, "R2", r2)

    # save txt
    out = np.column_stack([
        ids,
        trues,
        preds,
        preds-trues,
        Y[i_te,0],  # true_v2
        Y[i_te,1],  # true_S0
        Y[i_te,2],  # true_mean_pT
        Y[i_te,3],  # true_impact
        Y[i_te,4]   # true_dNch_deta  ← ADD THIS
    ])

    np.savetxt(
        f"{target_name}_predictions.txt",
        out,
        header="eventID true_target pred_target residual true_v2 true_S0 true_mean_pT true_impact true_dNch_deta"
    )

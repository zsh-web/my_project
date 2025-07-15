import torch
from torch.utils.data import DataLoader
import numpy as np
import os
from sklearn.metrics import confusion_matrix, accuracy_score
import time

def train_one_epoch(net, loader, optimizer, criterion, device):
    net.train()
    running_loss = 0
    all_preds = []
    all_labels = []

    for batch in loader:
        data = batch['feature'].float().to(device)
        labels = batch['label'].squeeze().long().to(device)

        optimizer.zero_grad()
        outputs = net(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    return avg_loss, acc, cm

def validate_one_epoch(net, loader, criterion, device):
    net.eval()
    running_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            data = batch['feature'].float().to(device)
            labels = batch['label'].squeeze().long().to(device)

            outputs = net(data)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    return avg_loss, acc, cm

def train_network(net, train_set, val_set, device, epochs=10, bs=20, optimizer=None, criterion=None, outdir=None, file_prefix=None):
    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=bs, shuffle=False)

    net.to(device)

    tr_losses, tr_accs = [], []
    val_losses, val_accs = [], []

    for epoch in range(epochs):
        start_time = time.time()

        tr_loss, tr_acc, tr_cm = train_one_epoch(net, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_cm = validate_one_epoch(net, val_loader, criterion, device)

        tr_losses.append(tr_loss)
        tr_accs.append(tr_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Epoch {epoch}:")
        print(f"  Train Loss: {tr_loss:.4f}, Train Acc: {tr_acc:.4f}")
        print(tr_cm)
        print(f"  Val   Loss: {val_loss:.4f}, Val   Acc: {val_acc:.4f}")
        print(val_cm)
        print(f"  Time: {time.time() - start_time:.2f} sec\n")

    if outdir and file_prefix:
        os.makedirs(outdir, exist_ok=True)
        torch.save(net.state_dict(), os.path.join(outdir, f"{file_prefix}_model.pth"))
        np.save(os.path.join(outdir, f"{file_prefix}_training_loss.npy"), np.array(tr_losses))
        np.save(os.path.join(outdir, f"{file_prefix}_validation_loss.npy"), np.array(val_losses))
        np.save(os.path.join(outdir, f"{file_prefix}_training_accuracy.npy"), np.array(tr_accs))
        np.save(os.path.join(outdir, f"{file_prefix}_validation_accuracy.npy"), np.array(val_accs))

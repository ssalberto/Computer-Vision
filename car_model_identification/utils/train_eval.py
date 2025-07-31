import torch
from tqdm import tqdm

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        correct, total, loss_sum = 0, 0, 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()
            loop.set_postfix(loss=loss_sum/(loop.n+1), acc=100.*correct/total)

        val_acc = evaluate_model(model, val_loader, device)
        scheduler.step(val_acc)
        print(f"Validation Acc: {val_acc:.2f}%")

def evaluate_model(model, data_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
    return 100. * correct / total

import torch
from torch import nn
from tqdm import tqdm
import os 
from args import args

criterion = nn.CrossEntropyLoss()

def train(model, train_loader, val_loader, optimizer, scheduler, criterion, num_epochs = args.epochs, model_name = "MODEL"):
    scaler = torch.amp.GradScaler('cuda')
    train_accuracies = []
    val_accuracies = []
    best_acc = 0.00

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1} of {num_epochs}")
        print("-" * 30)

        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc = "Training"):
            images, labels = images.to(args.device, non_blocking = True), labels.to(args.device, non_blocking = True)

            optimizer.zero_grad()

            with torch.amp.autocast(device_type='cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)

            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels).item()
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        train_accuracies.append(epoch_acc)
        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        model.eval()
        correct, total = 0, 0


        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc = "Validation"):
                images, labels = images.to(args.device, non_blocking = True), labels.to(args.device, non_blocking = True)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                correct += torch.sum(preds == labels).item()
                total += labels.size(0)

        val_acc = correct / total
        val_accuracies.append(val_acc)
        print(f"Val accuracy : {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join('./saved_models', f'{model_name}.pth'))
            print(f"Saved new best model with accuracy: {best_acc:.4f}")
        scheduler.step(val_acc)

    return model, train_accuracies, val_accuracies
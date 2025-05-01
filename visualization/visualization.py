import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import time
import torch
from args import args
from datasets.caltech import caltech

class_names = caltech.categories

def visualize_predictions(model, dataloader, class_names, num_images = 5):
    model.eval()
    images_shown = 0
    plt.figure(figsize = (15 , 15))

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(args.device), labels.to(args.device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            for i in range(images.size(0)):
                if images_shown >= num_images:
                    break
                img = images[i].cpu().permute(1,2,0) * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
                img = img.numpy().clip(0,1)
                plt.subplot(1, num_images, images_shown + 1)
                plt.imshow(img)
                plt.title(f"Pred: {class_names[preds[i]]}\nTrue: {class_names[labels[i]]}")
                plt.axis('off')
                images_shown += 1
            if images_shown >= num_images:
                break
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(model, dataloader, class_names):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(args.device), labels.to(args.device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    _, ax = plt.subplots(figsize=(25, 25))
    disp.plot(ax=ax, cmap='Blues')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.title("Confusion Matrix")
    plt.show()

def plot_learning_curve(train_accs, val_accs, label):
    epochs = range(1, len(train_accs) + 1)
    plt.plot(epochs, train_accs, 'bo-', label=f'{label} Train Acc')
    plt.plot(epochs, val_accs, 'ro-', label=f'{label} Val Acc')
    plt.title(f"Learning Curve - {label}")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()

def model_stats(model, dataloader):
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters: {param_count:,}")

    model.eval()
    start = time.time()
    with torch.no_grad():
        for i, (images, _) in enumerate(dataloader):
            images = images.to(args.device)
            _ = model(images)
            if i == 10:
                break
    end = time.time()
    avg_time = (end - start) / 10
    print(f"Approx. Inference Time per Batch: {avg_time:.4f} seconds")
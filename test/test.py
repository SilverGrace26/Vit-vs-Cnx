import torch 
from tqdm import tqdm
from args import args

def test(test_loader, model):

    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc = "Testing"):
            images, labels = images.to(args.device, non_blocking = True), labels.to(args.device, non_blocking = True)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels).item()
            total += labels.size(0)

    test_acc = correct/total
    print(f"Test accuracy : {test_acc:.4f}")
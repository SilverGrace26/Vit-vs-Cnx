from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
from datasets.datasetFromSubset import DatasetFromSubset
from torch.utils.data import DataLoader, Subset
from args import args

INPUT_SIZE = args.input_size
BATCH_SIZE = args.batch_size

train_transforms = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness = 0.2, contrast = 0.3, saturation = 0.2, hue = 0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


caltech = datasets.Caltech101(root = "./data", download = True)

indices = list(range(len(caltech)))
targets = [target for _, target in caltech]

train_idx, test_idx = train_test_split(indices, test_size = 0.2, stratify = targets)
train_idx, val_idx = train_test_split(train_idx, test_size = 0.3, stratify = [targets[i] for i in train_idx])

train_dataset = Subset(caltech, train_idx)
val_dataset = Subset(caltech, val_idx)
test_dataset = Subset(caltech, test_idx)

train_set = DatasetFromSubset(train_dataset, transforms = train_transforms)
val_set = DatasetFromSubset(val_dataset, transforms = test_transforms)
test_set = DatasetFromSubset(test_dataset, transforms = test_transforms)

train_loader = DataLoader(train_set, batch_size = BATCH_SIZE, shuffle = True, num_workers=4, pin_memory=True, prefetch_factor = 4, persistent_workers = True)
val_loader = DataLoader(val_set, batch_size = BATCH_SIZE, shuffle = False, num_workers=4, pin_memory=True, prefetch_factor = 4, persistent_workers = True)
test_loader = DataLoader(test_set, batch_size = BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, prefetch_factor = 4, persistent_workers = True)
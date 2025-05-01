import argparse
import torch

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--num_classes', type=int, default=102)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=3)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser.set_defaults(device=device)

    return parser.parse_args()

args = _parse_args()

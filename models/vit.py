import timm
from torch import optim
from args import args

vit_model = timm.create_model('vit_tiny_patch16_224', pretrained = True, num_classes = args.num_classes)
vit_model.to(args.device)

vit_optimizer = optim.AdamW(vit_model.parameters(), lr=args.lr, weight_decay = args.wd)
vit_scheduler = optim.lr_scheduler.ReduceLROnPlateau(vit_optimizer, mode = 'max', patience = 2, factor = 0.5)

import timm
from torch import optim
from args import args

cnx_model = timm.create_model('convnext_tiny', pretrained = True, num_classes = args.num_classes)
cnx_model.to(args.device)

cnx_optimizer = optim.AdamW(cnx_model.parameters(), lr=args.lr, weight_decay = args.wd)
cnx_scheduler = optim.lr_scheduler.ReduceLROnPlateau(cnx_optimizer, mode = 'max', patience = 2, factor = 0.5)
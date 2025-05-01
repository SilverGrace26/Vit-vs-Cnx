import torch
import os
from seed import set_seed
from models import cnx, vit 
from args import args
from train.train import train, criterion
from test.test import test
from datasets.caltech import train_loader, val_loader, test_loader
from visualization.visualization import class_names, visualize_predictions, plot_confusion_matrix, plot_learning_curve, model_stats


def main():

    set_seed(56)

    os.makedirs('./saved_models', exist_ok=True)

    vit_model, vit_train_accuracies, vit_val_accuracies = train(vit.vit_model, train_loader, val_loader, vit.vit_optimizer, vit.vit_scheduler, criterion, num_epochs = args.epochs, model_name = "vit_model")
    cnx_model, cnx_train_accuracies, cnx_val_accuracies = train(cnx.cnx_model, train_loader, val_loader, cnx.cnx_optimizer, cnx.cnx_scheduler, criterion, num_epochs = args.epochs, model_name = "cnx_model")

    vit_model = vit.vit_model
    vit_model.load_state_dict(torch.load('saved_models/vit_model.pth'))

    cnx_model = cnx.cnx_model
    cnx_model.load_state_dict(torch.load('saved_models/cnx_model.pth'))

    test(test_loader, vit_model)
    test(test_loader, cnx_model)

    visualize_predictions(vit_model, test_loader, class_names)
    visualize_predictions(cnx_model, test_loader, class_names)

    plot_confusion_matrix(vit_model, test_loader, class_names)
    plot_confusion_matrix(cnx_model, test_loader, class_names)

    model_stats(vit_model, test_loader)
    model_stats(cnx_model, test_loader)

    plot_learning_curve(vit_train_accuracies, vit_val_accuracies, label="ViT")
    plot_learning_curve(cnx_train_accuracies, cnx_val_accuracies, label="ConvNeXt")


if __name__ == '__main__':
    main()






























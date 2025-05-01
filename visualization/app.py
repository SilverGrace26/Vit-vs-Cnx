import gradio as gr
from PIL import Image
import torch
from numpy import ndarray
from args import args
from datasets.caltech import test_transforms
from models.vit import vit_model
from models.cnx import cnx_model
from visualization import class_names

def inference_example(image):
    if isinstance(image, ndarray):
        image = Image.fromarray(image)

    image_tensor = test_transforms(image).unsqueeze(0)
    image_tensor = image_tensor.to(args.device)

    vit_model.eval()
    cnx_model.eval()

    with torch.no_grad():
        vit_logits = vit_model(image_tensor)
        cnx_logits = cnx_model(image_tensor)

        vit_probs = torch.softmax(vit_logits, dim=1).cpu().numpy()[0]
        cnx_probs = torch.softmax(cnx_logits, dim=1).cpu().numpy()[0]

    vit_top5_idx = vit_probs.argsort()[-5:][::-1]
    cnx_top5_idx = cnx_probs.argsort()[-5:][::-1]

    vit_top5 = {class_names[i]: float(vit_probs[i]) for i in vit_top5_idx}
    cnx_top5 = {class_names[i]: float(cnx_probs[i]) for i in cnx_top5_idx}

    return vit_top5, cnx_top5

im = gr.Image(image_mode='RGB', label="Upload Image")
vit_label = gr.Label(label="ViT Predictions", num_top_classes=5)
cnx_label = gr.Label(label="ConvNeXt Predictions", num_top_classes=5)

iface = gr.Interface(fn=inference_example, inputs=im, outputs=[vit_label, cnx_label])
iface.launch(share=True)
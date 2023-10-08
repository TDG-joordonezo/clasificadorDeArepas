import os
import onnx
import onnxruntime
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

absolute_path = os.path.dirname(__file__)
model_path = os.path.join(absolute_path, "clasificadorArepas.onnx")
onnx_model = onnx.load(model_path)
onnx.checker.check_model(onnx_model)
ort_session = ort_session = onnxruntime.InferenceSession(
    model_path, providers=['AzureExecutionProvider', 'CPUExecutionProvider'])


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def preprocess_image(image_path):
    transform = transforms.Compose([
        # transforms.RandomHorizontalFlip(),  # Volteo horizontal aleatorio
        transforms.RandomVerticalFlip(),
        # transforms.RandomRotation(degrees=5),  # Rotación aleatoria de hasta 15 grados
        # transforms.RandomAffine(degrees=0, translate=(0.1, 0)),  # Afinamiento aleatorio
        transforms.Resize((120, 160)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x / 255.0)
    ])
    image = Image.open(image_path)
    image = transform(image)
    image = image.unsqueeze(0)

    return image


def predict_image(onnx_model_path, image_path):
    # ort_session = onnxruntime.InferenceSession(onnx_model_path)
    # ort_session = onnxruntime.InferenceSession(model_path, providers=['AzureExecutionProvider', 'CPUExecutionProvider'])

    image = preprocess_image(image_path)
    image_np = image.numpy()

    ort_inputs = {ort_session.get_inputs()[0].name: image_np}
    ort_outs = ort_session.run(None, ort_inputs)

    prediction = ort_outs[0][0]

    if prediction >= 0.5:
        return "Arepa Perfecta"+str(prediction)
    else:
        return "Arepa Defectuosa"+str(prediction)


onnx_model_path = "clasificadorArepas.onnx"
image_path = "arepasDataset/perfectas\IMG_20230820_170218625.jpg"
img_path = os.path.join(absolute_path, image_path)
result = predict_image(onnx_model_path, img_path)
print(f"Predicción para la imagen {image_path}: {result}")

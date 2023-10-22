import os
import torch
from torch import nn
from torch import optim
import torch.onnx
import torch.nn.functional as F
from torchvision.io import read_image
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import onnx
import onnxruntime
from sklearn.model_selection import train_test_split

relative_path = 'arepasDatasetSinFondo/defectuosas/IMG_20231014_173728759.png'
absolute_path = os.path.dirname(__file__)
full_path = os.path.join(absolute_path, relative_path)
img = read_image(full_path)
plt.imshow(img.T)
plt.show()
img.shape
img.dtype
img[:, 100:102, 100:102]

train_data_augmentation_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.Resize((240, 320)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x / 255.0)
])

dataset_path = './arepasDatasetSinFondo'
dataset = ImageFolder(
    dataset_path, transform=train_data_augmentation_transform)

num_classes = 2
dataset.targets = F.one_hot(torch.tensor(
    dataset.targets), num_classes=num_classes).float()

# Divide los datos en conjuntos de entrenamiento (80%) y prueba (20%)
train_data, test_data = train_test_split(
    dataset, test_size=0.2, random_state=42)

# Divide los datos de entrenamiento en conjuntos de entrenamiento (80%) y validación (20%)
train_data, val_data = train_test_split(
    train_data, test_size=0.2, random_state=42)

batch_size = 64

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)


# val_loader.dataset.targets.shape
"""
for batch in train_loader:
    images, labels = batch

    print("Image Shape", images.shape)
    print("Image Tensor", images[0][:, 100:102, 100:102])
    plt.imshow(images[0].T, cmap='gray')
    break
plt.show()
"""

def plot_loss(train_losses, test_losses):
    plt.figure(figsize=(12, 4))
    plt.plot(train_losses, label='Pérdida de entrenamiento', color='blue')
    plt.plot(test_losses, label='Pérdida de prueba', color='red')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.title('Curva de Pérdida')
    plt.grid(True)
    plt.show()


def plot_accuracy(train_accuracies, test_accuracies):
    plt.figure(figsize=(12, 4))
    plt.plot(train_accuracies, label='Precisión de entrenamiento', color='blue')
    plt.plot(test_accuracies, label='Precisión de prueba', color='red')
    plt.xlabel('Época')
    plt.ylabel('Precisión (%)')
    plt.legend()
    plt.title('Curva de Precisión')
    plt.grid(True)
    plt.show()


model = nn.Sequential(
    # nn.Conv2d(1, 32, kernel_size=3, padding=1),
    # nn.ReLU(),
    # nn.MaxPool2d(kernel_size=2, stride=2),

    # nn.Conv2d(32, 64, kernel_size=3, padding=1),
    # nn.ReLU(),
    # nn.MaxPool2d(kernel_size=2, stride=2),

    nn.Conv2d(1, 128, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    nn.Flatten(start_dim=1),
    nn.Linear(128 * 30 * 40, 100),
    nn.ReLU(),

    nn.Linear(100, 50),
    nn.ReLU(),

    nn.Dropout(p=0.5),

    nn.Linear(50, 1),
    nn.Sigmoid()

)
print(model)

loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


def train(model, num_epochs, train_dl, valid_dl):
    loss_hist_train = [0] * num_epochs
    acurracy_hist_train = [0] * num_epochs
    loss_hist_valid = [0] * num_epochs
    acurracy_hist_valid = [0] * num_epochs

    for epoch in range(num_epochs):
        model.train()
        for x_batch, y_batch in train_dl:
            pred = model(x_batch)[:, 0]
            loss = loss_fn(pred, y_batch.float())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_hist_train[epoch] += loss.item()*y_batch.size(0)
            is_correct = ((pred >= 0.5).float() == y_batch).float()
            acurracy_hist_train[epoch] += is_correct.sum()
        loss_hist_train[epoch] /= len(train_dl.dataset)
        acurracy_hist_train[epoch] /= len(train_dl.dataset)

        model.eval()
        with torch.no_grad():
            for x_batch, y_batch in valid_dl:
                pred = model(x_batch)[:, 0]
                loss = loss_fn(pred, y_batch.float())
                loss_hist_valid[epoch] += loss.item()*y_batch.size(0)
                is_correct = ((pred >= 0.5).float() == y_batch).float()
                acurracy_hist_valid[epoch] += is_correct.sum()
        loss_hist_valid[epoch] /= len(valid_dl.dataset)
        acurracy_hist_valid[epoch] /= len(valid_dl.dataset)

        print(f'Epoch [{epoch+1}/{num_epochs}] accuracy_train: '
              f'{acurracy_hist_train[epoch]:.4f} accuracy_test: '
              f'{acurracy_hist_valid[epoch]:.4f}')
    return loss_hist_train, loss_hist_valid, acurracy_hist_train, acurracy_hist_valid


num_epochs = 24
train_losses, test_losses, train_accuracies, test_accuracies = train(
    model, num_epochs, train_loader, test_loader)

# Define el dispositivo (GPU si está disponible, de lo contrario, CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Mueve tu modelo y datos al dispositivo
model.to(device)
# Definir la función de prueba


def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.binary_cross_entropy(
                output[:, 0], target.float(), reduction='sum').item()
            pred = output.round()
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100.0 * correct / len(test_loader.dataset)

    return test_loss, accuracy


plot_loss(train_losses, test_losses)
plot_accuracy(train_accuracies, test_accuracies)
test_loss, test_accuracy = test(model, test_loader)
print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')

# importar modelo
x = torch.rand(64, 1, 240, 320)
y = model.cpu()(x)

torch.onnx.export(model,
                  x,
                  "clasificadorArepas.onnx",
                  export_params=True,
                  opset_version=10,
                  do_constant_folding=True,
                  input_names=['input'],
                  output_names=['output'],
                  dynamic_axes={
                      'input': {
                          0: 'batch_size'
                      },
                      'output': {
                          0: 'batch_size'
                      }
                  })


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


ort_session = ort_session = onnxruntime.InferenceSession(
    "clasificadorArepas.onnx")
# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
ort_outs = ort_session.run(None, ort_inputs)

# compare ONNX Runtime and PyTorch results
np.testing.assert_allclose(to_numpy(x), ort_outs[0], rtol=1e-03, atol=1e-05)

print("Exported model has been tested with ONNXRuntime, and the result looks good!")

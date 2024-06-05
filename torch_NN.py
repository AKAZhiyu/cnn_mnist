import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset
from PIL import Image
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f'Training on device: {device}.')


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            # First Convolution Block
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.4),

            # Second Convolution Block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.4)
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3136, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.4),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

    def predict_image(self, image_path):
        """
        Perform image prediction, including image loading, preprocessing, and determining if color inversion is needed.

        Parameters:
        - image_path: File path to the image.

        Returns:
        - predicted_label: The predicted category.
        """
        # Define image transformation
        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # Load and convert image
        image = Image.open(image_path)
        image = image.convert('L')  # Ensure the image is grayscale

        # Determine if color inversion is needed
        array = np.array(image)
        mean_value = np.mean(array)
        if mean_value > 128:
            image = Image.fromarray(np.uint8(255 - array))  # Invert colors

        # Apply preprocessing
        input_tensor = transform(image)
        input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension

        # Automatically determine the device from model parameters
        device = next(self.parameters()).device

        # Transfer data and model to the detected device
        input_tensor = input_tensor.to(device)

        # Make a prediction
        self.eval()
        with torch.no_grad():
            predictions = self(input_tensor)
            _, predicted_label = torch.max(predictions, 1)

        return predicted_label.item()

    def predict_image_from_array(self, img_array):
        """
        Perform image prediction from a preprocessed numpy array directly.

        Parameters:
        - img_array: Preprocessed image as a numpy array.

        Returns:
        - predicted_label: The predicted category.
        """
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # # Convert numpy array to PIL Image for consistent preprocessing
        # img = Image.fromarray(np.uint8(img_array * 255)).convert('L')

        img_array = np.reshape(img_array, (28, 28))
        input_tensor = transform(img_array).unsqueeze(0).float()  # Add batch dimension

        # Determine the device automatically
        device = next(self.parameters()).device

        input_tensor = input_tensor.to(device)
        self.to(device)

        # Make a prediction
        self.eval()
        with torch.no_grad():
            predictions = self(input_tensor)
            _, predicted_label = torch.max(predictions, 1)

        return predicted_label.item()

if __name__ == "__main__":
    model = CNN().to(device)
    # 加载保存的参数
    model.load_state_dict(torch.load('model_epoch_40.pth'))

    # 将模型设置为评估模式
    model.eval()
    predicted_label = model.predict_image('test.png')
    print("Predicted label:", predicted_label)
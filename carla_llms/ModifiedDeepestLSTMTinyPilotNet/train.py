import numpy as np
import matplotlib.pyplot as plt
import torch
import matplotlib.pyplot as plt
from utils.load_dataset import CARLADataset
from torchvision.transforms import Compose
from utils.preprocess import FilterClassesTransform, ShiftAndAdjustSteer
from utils.ModifiedDeepestLSTMTinyPilotNet import PilotNetEmbeddingNoLight, PilotNetOneHot, PilotNetOneHot, PilotNetOneHotNoLight
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


def plot_histogram(dataset):
    throttle_vals = []
    brake_vals = []
    steer_vals = []
    speed_vals = []

    for images, speed, hlc, light, controls, distance in dataset:
        throttle_vals.append(controls[0].item())
        brake_vals.append(controls[2].item())
        steer_vals.append(controls[1].item())
        speed_vals.append(speed.item())

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    axs[0, 0].hist(throttle_vals, bins=100)
    axs[0, 0].set_title('Throttle')

    axs[0, 1].hist(brake_vals, bins=100)
    axs[0, 1].set_title('Brake')

    axs[1, 0].hist(steer_vals, bins=100)
    axs[1, 0].set_title('Steer')

    axs[1, 1].hist(speed_vals, bins=100)
    axs[1, 1].set_title('Speed')

    plt.show()


def load_data(batch_size, dataset_path, transform, one_hot, combined_control):
    train_data_dir = f"{dataset_path}"
    val_data_dir = f"{dataset_path}"

    train_dataset = CARLADataset(train_data_dir, transform=transform, one_hot=one_hot, combined_control=combined_control)
    val_dataset = CARLADataset(val_data_dir, transform=transform, one_hot=one_hot, combined_control=combined_control)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    return train_dataloader, val_dataloader


def train_model(model, train_dataloader, val_dataloader, epochs, device, lr=0.001):
    # loss function and optimizer
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()

        running_loss = 0.0

        for i, data in enumerate(train_dataloader):
            img, speed, hlc, light, controls = data
            img = img.to(device)
            speed = speed.to(device)
            hlc = hlc.to(device)
            light = light.to(device)
            controls = controls.to(device)

            outputs = model(img, speed, hlc, light)
            loss = criterion(outputs, controls)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 200 == 199:
                print(f'Epoch: {epoch + 1}, Batch: {i + 1}')

        train_loss = running_loss / len(train_dataloader)
        train_losses.append(train_loss)

        # validation
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(val_dataloader):
                img, speed, hlc, light, controls = data
                img = img.to(device)
                speed = speed.to(device)
                hlc = hlc.to(device)
                light = light.to(device)
                controls = controls.to(device)

                outputs = model(img, speed, hlc, light)
                loss = criterion(outputs, controls)

                running_loss += loss.item()

        val_loss = running_loss / len(val_dataloader)
        val_losses.append(val_loss)

        print(f'Epoch {epoch+1}, Train Loss: {train_loss}, Val Loss: {val_loss}')
        torch.save(model.state_dict(), f'model_checkpoint_{epoch}.pth')

    torch.save({'train_losses': train_losses, 'val_losses': val_losses}, 'losses.pth')


filter_transform = FilterClassesTransform(mode='both', classes_to_keep=[1, 7, 12, 13, 14, 15, 16, 17, 18, 19, 24])  # keep traffic light
shift_transform = ShiftAndAdjustSteer(shift_fraction=0.1, steer_adjust=1.0)
transforms = Compose([filter_transform, shift_transform])

# Set the batch size and number of epochs
batch_size = 32
epochs = 20

transform = None
one_hot = True
combined_control = True

train_dataloader, val_dataloader = load_data(batch_size, '../data/', transform=transforms, one_hot=one_hot, combined_control=combined_control)
print("loaded data")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = PilotNetOneHot((288, 200, 6), 2, 4, 4)  # combined control
model.to(device)
model_dict = torch.load(f'./models/v10.0.pth')
model.load_state_dict(model_dict)
print(model_dict)

train_model(model, train_dataloader, val_dataloader, epochs, device, lr=0.001)

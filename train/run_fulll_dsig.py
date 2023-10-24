"""Training script for training the weather forecasting model"""
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../")

from glob import glob
import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms as transforms
import xarray as xr
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from graph_weather import GraphWeatherForecaster
from graph_weather.data import const
from graph_weather.models.losses import NormalizedMSELoss
import matplotlib.pyplot as plt
import seaborn as sns


class XrDataset(Dataset):
    def __init__(self, file_name):
        super().__init__()

        self.data = xr.open_dataset(file_name, engine="netcdf4")

    def __len__(self):
        return len(self.data.time) - 1

    def __getitem__(self, idx):
        # start_idx = np.random.randint(0, len(self.data.time) - 1)
        data = self.data.isel(time=slice(idx, idx + 2))
        start = data.isel(time=0)
        end = data.isel(time=1)

        # if inter_data is not None and start != inter_data:
        #     start = inter_data
        #     end = data.isel(time=0)
        # elif start == inter_data:
        #     start = data.isel(time = 0)
        #     end = data.isel(time = 1)
        # else:
        #     start = data.isel(time=0)
        #     try:
        #         end = data.isel(time=1)
        #     except IndexError:
        #         inter_data = data.isel(time=0)

        # Stack the data into a large data cube
        input_data = np.stack(
            [
                (start[f"{var}"].values - const.FORECAST_MEANS[f"{var}"])
                / (const.FORECAST_STD[f"{var}"] + 0.0001)
                for var in start.data_vars
            ],
        )
        # input_data = np.stack(
        #     [(start[f"{var}"].values) for var in start.data_vars], axis=-1
        # )
        input_data = np.nan_to_num(input_data)

        assert not np.isnan(input_data).any()
        output_data = np.stack(
            [
                (end[f"{var}"].values - const.FORECAST_MEANS[f"{var}"])
                / (const.FORECAST_STD[f"{var}"] + 0.0001)
                for var in end.data_vars
            ]
        )
        # output_data = np.stack(
        #     [(end[f"{var}"].values) for var in end.data_vars], axis=-1
        # )
        output_data = np.nan_to_num(output_data)
        assert not np.isnan(output_data).any()
        transform = transforms.Compose([transforms.ToTensor()])
        # Normalize now
        return (
            transform(input_data)
            .transpose(0, 2)
            .transpose(1, 2)
            .reshape(-1, input_data.shape[0]),
            transform(input_data)
            .transpose(0, 2)
            .transpose(1, 2)
            .reshape(-1, input_data.shape[0]),
        )


data = xr.open_dataset(
    "../graph_weather/data/MERRA2_400.inst3_3d_asm_Nv.20230701_merged.nc",
    engine="netcdf4",
)
# print(data)
# print("Done coarsening")
# meshgrid takes in the lat and lon values, creates the arrays of every single latitude with the length of every single longitude, np.array reshapes it to get a matrix
# which contains the lat lon co-related values, eg. 90*180, 90*179.375 etc.
lat_lons = np.array(np.meshgrid(data.lat.values, data.lon.values)).T.reshape(-1, 2)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "mps" if torch.backends.mps.is_available() else "cpu"
# Get the variance of the variables
feature_variances = []
for var in data.data_vars:
    feature_variances.append(const.FORECAST_DIFF_STD[var] ** 2)
criterion = NormalizedMSELoss(
    lat_lons=lat_lons, feature_variance=feature_variances, device=device
).to(device)
means = []
# dataset = DataLoader(XrDataset(), batch_size=1)
# files_dataloader = DataLoader(FileDataset("graph_weather/data/train_data"), batch_size = 1)
model = GraphWeatherForecaster(lat_lons, feature_dim=65, num_blocks=6).to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.000001)
print("Done Setup")
import time

train_files = glob(
    "../graph_weather/data/train_data/*.nc", recursive=True
)
val_files = glob(
    "../graph_weather/data/val_data/*.nc", recursive=True
)
running_loss, running_val_loss = [], []
for epoch in range(20):  # loop over the dataset multiple times
    model.train()
    start = time.time()
    inter_data = None
    running_loss_files = []
    running_val_loss_files = []
    for name in train_files:
        dataset = DataLoader(XrDataset(name), batch_size=1)

        # print(f"Start Epoch: {epoch+1}")
        for i, data in tqdm(enumerate(dataset), total=len(dataset), leave=False):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss_files.append(loss.item())
        print(f"{epoch + 1} training_loss: {np.mean(running_loss_files)}")

    model.eval()
    for name in val_files:
        dataset = DataLoader(XrDataset(name), batch_size=1)

        # print(f"Start Epoch: {epoch+1}")
        for i, data in tqdm(enumerate(dataset), total=len(dataset), leave=False):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # forward + backward + optimize
            outputs = model(inputs)

            val_loss = criterion(outputs, labels)

            # print statistics
            running_val_loss_files.append(val_loss.item())
            # print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / (i + 1):.3f}")
        print(f"{epoch + 1} validation_loss: {np.mean(running_val_loss_files)}")

    running_loss.append(np.mean(running_loss_files))
    running_val_loss.append(np.mean(running_val_loss_files))
end = time.time()
print(f"Time: {end - start} sec")
plt.plot(running_loss)
plt.plot(running_val_loss)
plt.title("model accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(["train", "val"], loc="upper left")
plt.savefig("openweather_20epochs_merra_batch_lr_0000001.png")
plt.show()
# if epoch % 5 == 0:
#     assert not np.isnan(running_loss)
#     model.push_to_hub(
#         "graph-weather-forecaster-2.0deg",
#         organization="openclimatefix",
#         commit_message=f"Add model Epoch={epoch}",
#     )


class TestXrDataset(Dataset):
    def __init__(self, file_name):
        super().__init__()

        self.data = xr.open_dataset(file_name, engine="netcdf4")

    def __len__(self):
        return len(self.data.time) - 1

    def __getitem__(self, idx):
        # start_idx = np.random.randint(0, len(self.data.time) - 1)
        data = self.data.isel(time=slice(idx, idx + 2))
        start = data.isel(time=0)
        end = data.isel(time=1)

        # if inter_data is not None and start != inter_data:
        #     start = inter_data
        #     end = data.isel(time=0)
        # elif start == inter_data:
        #     start = data.isel(time = 0)
        #     end = data.isel(time = 1)
        # else:
        #     start = data.isel(time=0)
        #     try:
        #         end = data.isel(time=1)
        #     except IndexError:
        #         inter_data = data.isel(time=0)

        # Stack the data into a large data cube
        input_data = np.stack(
            [(start[f"{var}"].values) for var in start.data_vars],
        )
        # input_data = np.stack(
        #     [(start[f"{var}"].values) for var in start.data_vars], axis=-1
        # )
        input_data = np.nan_to_num(input_data)

        assert not np.isnan(input_data).any()
        output_data = np.stack([(end[f"{var}"].values) for var in end.data_vars])
        # output_data = np.stack(
        #     [(end[f"{var}"].values) for var in end.data_vars], axis=-1
        # )
        output_data = np.nan_to_num(output_data)
        assert not np.isnan(output_data).any()
        transform = transforms.Compose([transforms.ToTensor()])
        # Normalize now
        return (
            transform(input_data)
            .transpose(0, 2)
            .transpose(1, 2)
            .reshape(-1, input_data.shape[0]),
            transform(input_data)
            .transpose(0, 2)
            .transpose(1, 2)
            .reshape(-1, input_data.shape[0]),
        )


for name in glob(
    "../graph_weather/data/test_data/*.nc", recursive=True
):
    dataset = DataLoader(TestXrDataset(name), batch_size=1)
    fig1, ax1 = plt.subplots(2, 2, figsize=(12, 12))
    fig1.suptitle("Test Image")
    for i, data in tqdm(enumerate(dataset), total=len(dataset), leave=False):
        # get the inputs; data is a list of [inputs, labels]
        inputs_test, labels = data[0].to(device), data[1].to(device)
        outputs_test = model(inputs_test)
        diff_test = labels - outputs_test
        fig1, ax1 = plt.subplots(2, 2, figsize=(12, 12))
        sns.heatmap(
            torch.reshape(inputs, (1, 361, 576, 65))[0, :, :, 27],
            cbar=True,
            cmap="Blues",
            ax=ax1[0][0],
        )
        ax1[0][0].set_title("Test Input Image")

        sns.heatmap(
            torch.reshape(labels, (1, 361, 576, 65))[0, :, :, 27],
            cbar=True,
            cmap="Blues",
            ax=ax1[0][1],
        )
        ax1[0][1].set_title("Test Output Image")

        sns.heatmap(
            torch.reshape(outputs_test, (1, 361, 576, 65))
            .detach()
            .numpy()[0, :, :, 27],
            cmap="Blues",
            cbar=True,
            ax=ax1[1][0],
        )
        ax1[1][0].set_title("Predicted Image")

        sns.heatmap(
            torch.reshape(diff_test, (1, 361, 576, 65)).detach().numpy()[0, :, :, 27],
            cmap="Blues",
            cbar=True,
            ax=ax1[1][1],
        )
        ax1[1][1].set_title("Difference in actual output and prediction")

        plt.savefig(f"results_{name.split('/')[-1][:-3]}_{i}_T_41.png")

        fig1, ax1 = plt.subplots(2, 2, figsize=(12, 12))
        sns.heatmap(
            torch.reshape(inputs, (1, 361, 576, 65))[0, :, :, 44],
            cbar=True,
            cmap="Blues",
            ax=ax1[0][0],
        )
        ax1[0][0].set_title("Test Input Image")

        sns.heatmap(
            torch.reshape(labels, (1, 361, 576, 65))[0, :, :, 44],
            cbar=True,
            cmap="Blues",
            ax=ax1[0][1],
        )
        ax1[0][1].set_title("Test Output Image")

        sns.heatmap(
            torch.reshape(outputs_test, (1, 361, 576, 65))
            .detach()
            .numpy()[0, :, :, 44],
            cmap="Blues",
            cbar=True,
            ax=ax1[1][0],
        )
        ax1[1][0].set_title("Predicted Image")

        sns.heatmap(
            torch.reshape(diff_test, (1, 361, 576, 65)).detach().numpy()[0, :, :, 44],
            cmap="Blues",
            cbar=True,
            ax=ax1[1][1],
        )
        ax1[1][1].set_title("Difference in actual output and prediction")

        plt.savefig(f"results_{name.split('/')[-1][:-3]}_{i}_U_45.png")


# for epoch in range(100):  # loop over the dataset multiple times
#     for j, file_data in tqdm(enumerate(files_dataloader), total=len(files_dataloader), leave=False):
#         dataset = DataLoader(XrDataset(file_data), batch_size=1)
#         running_loss = 0.0
#         start = time.time()
#         print(f"Start Epoch: {epoch}")
#         for i, data in tqdm(enumerate(dataset), total=len(dataset), leave=False):
#             # get the inputs; data is a list of [inputs, labels]
#             inputs, labels = data[0].to(device), data[1].to(device)
#             # zero the parameter gradients
#             optimizer.zero_grad()

# forward + backward + optimize
#             outputs = model(inputs)

#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             # print statistics
#             running_loss += loss.item()
#     end = time.time()
#     print(
#         f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / (i + 1):.3f} Time: {end - start} sec")
# if epoch % 5 == 0:
#     assert not np.isnan(running_loss)
#     model.push_to_hub(
#         "graph-weather-forecaster-2.0deg",
#         organization="openclimatefix",
#         commit_message=f"Add model Epoch={epoch}",
#     )

# print("Finished Training")

# def plot_data(
#     data: dict[str, xr.Dataset],
#     fig_title: str,
#     plot_size: float = 5,
#     robust: bool = False,
#     cols: int = 4
#     ) -> tuple[xr.Dataset, matplotlib.colors.Normalize, str]:

#   first_data = next(iter(data.values()))[0]
#   max_steps = first_data.sizes.get("time", 1)
#   assert all(max_steps == d.sizes.get("time", 1) for d, _, _ in data.values())

#   cols = min(cols, len(data))
#   rows = math.ceil(len(data) / cols)
#   figure = plt.figure(figsize=(plot_size * 2 * cols,
#                                plot_size * rows))
#   figure.suptitle(fig_title, fontsize=16)
#   figure.subplots_adjust(wspace=0, hspace=0)
#   figure.tight_layout()

#   images = []
#   for i, (title, (plot_data, norm, cmap)) in enumerate(data.items()):
#     ax = figure.add_subplot(rows, cols, i+1)
#     ax.set_xticks([])
#     ax.set_yticks([])
#     ax.set_title(title)
#     im = ax.imshow(
#         plot_data.isel(time=0, missing_dims="ignore"), norm=norm,
#         origin="lower", cmap=cmap)
#     plt.colorbar(
#         mappable=im,
#         ax=ax,
#         orientation="vertical",
#         pad=0.02,
#         aspect=16,
#         shrink=0.75,
#         cmap=cmap,
#         extend=("both" if robust else "neither"))
#     images.append(im)


# %%

# input_data = np.stack(
#     [
#         (start[f"{var}"].values - const.FORECAST_MEANS[f"{var}"])
#         / (const.FORECAST_STD[f"{var}"] + 0.0001)
#         for var in start.data_vars
#         if "mb" in var or "surface" in var
#     ],
#     axis=-1,
# )

# for var in start.data_vars:
#     if "mb" in var or "surface" in var:
#         temp_data = np.stack(
#             [
#                 (start[f"{var}"].values - const.FORECAST_MEANS[f"{var}"])
#                 / (const.FORECAST_STD[f"{var}"] + 0.0001)
#             ],
#             axis=-1,
#         )

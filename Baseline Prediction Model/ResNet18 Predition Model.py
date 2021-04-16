# import packages
import os, gc
import warnings

import matplotlib
import zarr
import numpy as np
import pandas as pd
from l5kit.evaluation.metrics import neg_multi_log_likelihood, time_displace
from tqdm import tqdm
from typing import Dict
from collections import Counter
from prettytable import PrettyTable

# level5 toolkit
from l5kit.data import PERCEPTION_LABELS
from l5kit.dataset import EgoDataset, AgentDataset
from l5kit.data import ChunkedDataset, LocalDataManager

# level5 toolkit
from l5kit.configs import load_config_data
from l5kit.geometry import transform_points
from l5kit.rasterization import build_rasterizer
from l5kit.visualization import draw_trajectory, draw_reference_trajectory, TARGET_POINTS_COLOR, PREDICTED_POINTS_COLOR
from l5kit.evaluation import write_pred_csv, compute_metrics_csv, read_gt_csv, create_chopped_dataset
from l5kit.evaluation.chop_dataset import MIN_FUTURE_STEPS
from pathlib import Path

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
from matplotlib import animation
from colorama import Fore, Back, Style

# deep learning
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.models.resnet import resnet18, resnet50, resnet34

import time

warnings.filterwarnings("ignore")

import torch
from torch import Tensor

# training cfg
training_cfg = {

    'format_version': 4,

    ## Model options
    'model_params': {
        'model_architecture': 'resnet50',
        'history_num_frames': 10,
        'history_step_size': 1,
        'history_delta_time': 0.1,
        'future_num_frames': 50,
        'future_step_size': 1,
        'future_delta_time': 0.1,
        'lr': 1e-3,
    },

    ## Input raster parameters
    'raster_params': {

        'raster_size': [224, 224],
        # raster's spatial resolution [meters per pixel]: the size in the real world one pixel corresponds to.
        'pixel_size': [0.5, 0.5],  # From 0 to 1 per axis, [0.5,0.5] would show the ego centered in the image.
        'ego_center': [0.25, 0.5],
        'map_type': "py_semantic",

        # the keys are relative to the dataset environment variable
        'satellite_map_key': "aerial_map/aerial_map.png",
        'semantic_map_key': "semantic_map/semantic_map.pb",
        'dataset_meta_key': "meta.json",

        # e.g. 0.0 include every obstacle, 0.5 show those obstacles with >0.5 probability of being
        # one of the classes we care about (cars, bikes, peds, etc.), >=1.0 filter all other agents.
        'filter_agents_threshold': 0.5
    },

    ## Data loader options
    'train_data_loader': {
        'key': "scenes/train.zarr",
        'batch_size': 10,
        'shuffle': True,
        'num_workers': 4
    },

    'val_data_loader': {
        'key': "scenes/validate.zarr",
        'batch_size': 10,
        'shuffle': False,
        'num_workers': 16
    },

    ## Train params
    'train_params': {
        'max_num_steps': 10000
    },
    'validate_params': {
        'max_num_steps': 100
    }
}


def build_model(cfg: Dict) -> torch.nn.Module:
    # load pre-trained Conv2D model
    model = resnet18(pretrained=True)

    # change input channels number to match the rasterizer's output
    num_history_channels = (training_cfg["model_params"]["history_num_frames"] + 1) * 2
    num_in_channels = 3 + num_history_channels
    model.conv1 = nn.Conv2d(
        num_in_channels,
        model.conv1.out_channels,
        kernel_size=model.conv1.kernel_size,
        stride=model.conv1.stride,
        padding=model.conv1.padding,
        bias=False,
    )
    # change output size to (X, Y) * number of future states
    num_targets = 2 * cfg["model_params"]["future_num_frames"]
    model.fc = nn.Linear(in_features=512, out_features=num_targets)

    return model


def forward(data, model, device, criterion):
    inputs = data["image"].to(device)
    target_availabilities = data["target_availabilities"].unsqueeze(-1).to(device)
    targets = data["target_positions"].to(device)
    # Forward pass
    outputs = model(inputs).reshape(targets.shape)
    loss = criterion(outputs, targets)
    # not all the output steps are valid, but we can filter them out from the loss using availabilities
    loss = loss * target_availabilities
    loss = loss.mean()
    return loss, outputs


DIR_INPUT = "/home/SharedStorage2/NewUsersDir/aledhari/npate181/Kaggle_data"

# set env variable for data
os.environ["L5KIT_DATA_FOLDER"] = DIR_INPUT
dm = LocalDataManager(None)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model = build_model(training_cfg).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss(reduction="none")

print("Model Loaded. Loading training dataset...")
print("==================================TRAIN DATA==================================")
# training cfg
train_cfg = training_cfg["train_data_loader"]

# rasterizer
rasterizer = build_rasterizer(training_cfg, dm)

# dataloader
train_zarr = ChunkedDataset(dm.require(train_cfg["key"])).open()
train_dataset = AgentDataset(training_cfg, train_zarr, rasterizer)
train_dataloader = DataLoader(train_dataset, shuffle=train_cfg["shuffle"], batch_size=train_cfg["batch_size"],
                              num_workers=train_cfg["num_workers"])
print(train_dataset)

# ==== TRAIN LOOP
tr_it = iter(train_dataloader)
progress_bar = tqdm(range(training_cfg["train_params"]["max_num_steps"]))
losses_train = []
for _ in progress_bar:
    try:
        data = next(tr_it)
    except StopIteration:
        tr_it = iter(train_dataloader)
        data = next(tr_it)
    model.train()
    torch.set_grad_enabled(True)
    loss, _ = forward(data, model, device, criterion)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses_train.append(loss.item())
    progress_bar.set_description(f"loss: {loss.item()} loss(avg): {np.mean(losses_train)}")

torch.save(model.state_dict(), 'baseline_model.pth')

evaluate_cfg = training_cfg["val_data_loader"]
evaluation_zarr = ChunkedDataset(dm.require(evaluate_cfg["key"])).open()
evaluation_dataset = AgentDataset(training_cfg, evaluation_zarr, rasterizer)
evaluation_dataloader = DataLoader(evaluation_dataset, shuffle=evaluate_cfg["shuffle"],
                                   batch_size=evaluate_cfg["batch_size"],
                                   num_workers=evaluate_cfg["num_workers"])
print(evaluation_dataset)
print(f"Length of evaluation : {len(evaluation_dataset)}")
print("Validation data loaded. Calculating validation loss...")
ev_it = iter(evaluation_dataloader)
progress_bar = tqdm(range(training_cfg["validate_params"]["max_num_steps"]))

losses_evaluation = []

for _ in progress_bar:
    try:
        data = next(tr_it)
    except StopIteration:
        tr_it = iter(evaluation_dataloader)
        data = next(tr_it)
    model.eval()
    torch.set_grad_enabled(False)
    validate_loss, _ = forward(data, model, device, criterion)

    losses_evaluation.append(validate_loss.item())
    progress_bar.set_description(f"loss: {validate_loss.item()} loss(avg): {np.mean(losses_evaluation)}")

# ===== GENERATE AND LOAD CHOPPED DATASET
num_frames_to_chop = 50
eval_cfg = training_cfg["val_data_loader"]
eval_base_path = create_chopped_dataset(dm.require(eval_cfg["key"]),
                                        training_cfg["raster_params"]["filter_agents_threshold"],
                                        num_frames_to_chop, training_cfg["model_params"]["future_num_frames"],
                                        MIN_FUTURE_STEPS)

eval_zarr_path = str(Path(eval_base_path) / Path(dm.require(eval_cfg["key"])).name)
eval_mask_path = str(Path(eval_base_path) / "mask.npz")
eval_gt_path = str(Path(eval_base_path) / "gt.csv")

eval_zarr = ChunkedDataset(eval_zarr_path).open()
eval_mask = np.load(eval_mask_path)["arr_0"]
# ===== INIT DATASET AND LOAD MASK
eval_dataset = AgentDataset(training_cfg, eval_zarr, rasterizer, agents_mask=eval_mask)
eval_dataloader = DataLoader(eval_dataset, shuffle=eval_cfg["shuffle"], batch_size=eval_cfg["batch_size"],
                             num_workers=eval_cfg["num_workers"])

model.eval()
torch.set_grad_enabled(False)

# store information for evaluation
future_coords_offsets_pd = []
timestamps = []
agent_ids = []

progress_bar = tqdm(eval_dataloader)
for data in progress_bar:
    _, ouputs = forward(data, model, device, criterion)

    # convert agent coordinates into world offsets
    agents_coords = ouputs.cpu().numpy()
    world_from_agents = data["world_from_agent"].numpy()
    centroids = data["centroid"].numpy()
    coords_offset = []

    for agent_coords, world_from_agent, centroid in zip(agents_coords, world_from_agents, centroids):
        coords_offset.append(transform_points(agent_coords, world_from_agent) - centroid[:2])

    future_coords_offsets_pd.append(np.stack(coords_offset))
    timestamps.append(data["timestamp"].numpy().copy())
    agent_ids.append(data["track_id"].numpy().copy())

pred_path = 'submission_2nd_test.csv'
write_pred_csv(pred_path,
               timestamps=np.concatenate(timestamps),
               track_ids=np.concatenate(agent_ids),
               coords=np.concatenate(future_coords_offsets_pd),
               )

metrics = compute_metrics_csv(eval_gt_path, pred_path, [neg_multi_log_likelihood, time_displace])
for metric_name, metric_mean in metrics.items():
    print(metric_name, metric_mean)

model.eval()
torch.set_grad_enabled(False)
gt_rows = {}
for row in read_gt_csv(eval_gt_path):
    gt_rows[row["track_id"] + row["timestamp"]] = row["coord"]

eval_ego_dataset = EgoDataset(training_cfg, eval_dataset.dataset, rasterizer)

for frame_number in range(99, len(eval_zarr.frames), 1000):  # start from last frame of scene_0 and increase by 100
    agent_indices = eval_dataset.get_frame_indices(frame_number)
    if not len(agent_indices):
        continue

    # get AV point-of-view frame
    data_ego = eval_ego_dataset[frame_number]
    im_ego = rasterizer.to_rgb(data_ego["image"].transpose(1, 2, 0))
    center = np.asarray(training_cfg["raster_params"]["ego_center"]) * training_cfg["raster_params"]["raster_size"]

    predicted_positions = []
    target_positions = []

    for v_index in agent_indices:
        data_agent = eval_dataset[v_index]

        out_net = model(torch.from_numpy(data_agent["image"]).unsqueeze(0).to(device))
        out_pos = out_net[0].reshape(-1, 2).detach().cpu().numpy()
        # store absolute world coordinates
        predicted_positions.append(transform_points(out_pos, data_agent["world_from_agent"]))
        # retrieve target positions from the GT and store as absolute coordinates
        track_id, timestamp = data_agent["track_id"], data_agent["timestamp"]
        target_positions.append(gt_rows[str(track_id) + str(timestamp)] + data_agent["centroid"][:2])

    # convert coordinates to AV point-of-view so we can draw them
    predicted_positions = transform_points(np.concatenate(predicted_positions), data_ego["raster_from_world"])
    target_positions = transform_points(np.concatenate(target_positions), data_ego["raster_from_world"])

    draw_trajectory(im_ego, predicted_positions, PREDICTED_POINTS_COLOR)
    draw_trajectory(im_ego, target_positions, TARGET_POINTS_COLOR)
    f3 = plt2.figure(v_index)
    plt2.imshow(im_ego[::-1])
    plt2.savefig("/home/SharedStorage2/NewUsersDir/aledhari/npate181/MachineVisionProject/Final_submission/Baseline_Predictions/baseline_prediction" + str(frame_number) + ".png")

graph_label = f"Training loss(avg): {np.mean(losses_train)}"
f1 = plt.figure(1)
plt.plot(losses_train, label=graph_label)
plt.title('Training loss Vs. Sample')
plt.xlabel('Training')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.savefig("/home/SharedStorage2/NewUsersDir/aledhari/npate181/MachineVisionProject/Final_submission/baseline_training.png")

graph_label = f"Validation loss(avg): {np.mean(losses_evaluation)}"
f2 = plt.figure(2)
plt.plot(losses_evaluation, label=graph_label)
plt.title('Validate loss Vs. Sample')
plt.xlabel('Validate')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.savefig("/home/SharedStorage2/NewUsersDir/aledhari/npate181/MachineVisionProject/Final_submission/Baseline_Predictions/validation_loss.png")

# import packages
import os, gc
import warnings

import matplotlib
import zarr
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict
from collections import Counter
from prettytable import PrettyTable

# level5 toolkit
from l5kit.data import PERCEPTION_LABELS
from l5kit.dataset import EgoDataset, AgentDataset
from l5kit.data import ChunkedDataset, LocalDataManager

# level5 toolkit
from l5kit.evaluation.metrics import neg_multi_log_likelihood, time_displace
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
        'batch_size': 16,
        'shuffle': True,
        'num_workers': 4
    },

    'val_data_loader': {
        'key': "scenes/validate.zarr",
        'batch_size': 12,
        'shuffle': False,
        'num_workers': 16
    },

    ## Train params
    'train_params': {
        'max_num_steps': 100000
    },
    'validate_params': {
        'max_num_steps': 10000
    }
}


def pytorch_neg_multi_log_likelihood_batch(
        gt: Tensor, pred: Tensor, confidences: Tensor, avails: Tensor
) -> Tensor:
    assert len(pred.shape) == 4, f"expected 3D (MxTxC) array for pred, got {pred.shape}"
    batch_size, num_modes, future_len, num_coords = pred.shape

    assert gt.shape == (batch_size, future_len, num_coords), f"expected 2D (Time x Coords) array for gt, got {gt.shape}"
    assert confidences.shape == (batch_size, num_modes), f"expected 1D (Modes) array for gt, got {confidences.shape}"
    assert torch.allclose(torch.sum(confidences, dim=1),
                          confidences.new_ones((batch_size,))), "confidences should sum to 1"
    assert avails.shape == (batch_size, future_len), f"expected 1D (Time) array for gt, got {avails.shape}"
    # assert all data are valid
    assert torch.isfinite(pred).all(), "invalid value found in pred"
    assert torch.isfinite(gt).all(), "invalid value found in gt"
    assert torch.isfinite(confidences).all(), "invalid value found in confidences"
    assert torch.isfinite(avails).all(), "invalid value found in avails"

    # convert to (batch_size, num_modes, future_len, num_coords)
    gt = torch.unsqueeze(gt, 1)  # add modes
    avails = avails[:, None, :, None]  # add modes and cords

    # error (batch_size, num_modes, future_len)
    error = torch.sum(((gt - pred) * avails) ** 2, dim=-1)  # reduce coords and use availability

    with np.errstate(divide="ignore"):  # when confidence is 0 log goes to -inf, but we're fine with it
        # error (batch_size, num_modes)
        error = torch.log(confidences) - 0.5 * torch.sum(error, dim=-1)  # reduce time

    # use max aggregator on modes for numerical stability
    # error (batch_size, num_modes)
    max_value, _ = error.max(dim=1, keepdim=True)  # error are negative at this point, so max() gives the minimum one
    error = -torch.log(torch.sum(torch.exp(error - max_value), dim=-1, keepdim=True)) - max_value  # reduce modes
    # print("error", error)
    return torch.mean(error)


def pytorch_neg_multi_log_likelihood_single(
        gt: Tensor, pred: Tensor, avails: Tensor
) -> Tensor:
    batch_size, future_len, num_coords = pred.shape
    confidences = pred.new_ones((batch_size, 1))
    return pytorch_neg_multi_log_likelihood_batch(gt, pred.unsqueeze(1), confidences, avails)


print("Sucessfully imported libraries. Loading model...")


class LyftModel(nn.Module):

    def __init__(self, cfg, num_modes=3):
        super().__init__()

        # set pretrained=True while training
        self.backbone = resnet50(pretrained=False)

        num_history_channels = (cfg["model_params"]["history_num_frames"] + 1) * 2
        num_in_channels = 3 + num_history_channels

        self.backbone.conv1 = nn.Conv2d(
            num_in_channels,
            self.backbone.conv1.out_channels,
            kernel_size=self.backbone.conv1.kernel_size,
            stride=self.backbone.conv1.stride,
            padding=self.backbone.conv1.padding,
            bias=False,
        )

        # This is 512 for resnet18 and resnet34;
        # And it is 2048 for the other resnets
        backbone_out_features = 2048

        # X, Y coords for the future positions (output shape: Bx50x2)
        future_len = cfg["model_params"]["future_num_frames"]
        num_targets = 2 * future_len

        # You can add more layers here.
        self.head = nn.Sequential(
            # nn.Dropout(0.2),
            nn.Linear(in_features=backbone_out_features, out_features=4096),
        )
        self.num_preds = num_targets * num_modes
        self.num_modes = num_modes

        self.logit = nn.Linear(4096, out_features=self.num_preds + num_modes)

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.head(x)
        x = self.logit(x)
        bs, _ = x.shape
        pred, confidences = torch.split(x, self.num_preds, dim=1)
        pred = pred.view(bs, self.num_modes, 50, 2)
        assert confidences.shape == (bs, self.num_modes)
        confidences = torch.softmax(confidences, dim=1)
        return pred, confidences


def forward(data, model, device, criterion=pytorch_neg_multi_log_likelihood_batch):
    inputs = data["image"].to(device)
    target_availabilities = data["target_availabilities"].to(device)
    targets = data["target_positions"].to(device)
    # Forward pass
    preds, confidences = model(inputs)
    loss = criterion(targets, preds, confidences, target_availabilities)
    return loss, preds, confidences


# root directory
DIR_INPUT = "/home/SharedStorage2/NewUsersDir/aledhari/npate181/Kaggle_data"

# set env variable for data
os.environ["L5KIT_DATA_FOLDER"] = DIR_INPUT
dm = LocalDataManager(None)

rasterizer = build_rasterizer(training_cfg, dm)
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
model = LyftModel(training_cfg).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

model.load_state_dict(torch.load("/home/SharedStorage2/NewUsersDir/aledhari/npate181/"
                                 "MachineVisionProject/Final_submission/final_submission.pth"))
print("Model Loaded. Loading Validation dataset...")

# dataloader
print("==================================VALIDATE DATA==================================")
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
        data = next(ev_it)
    except StopIteration:
        ev_it = iter(evaluation_dataset)
        data = next(ev_it)
    model.eval()
    torch.set_grad_enabled(False)

    # forward pass
    loss_evaluation, _, _ = forward(data, model, device)
    losses_evaluation.append(loss_evaluation.item())
    progress_bar.set_description(f"loss: {loss_evaluation.item()} loss(avg): {np.mean(losses_evaluation)}")
print(f"loss: {loss_evaluation.item()} loss(avg): {np.mean(losses_evaluation)}")

graph_label = f"Validation loss(avg): {np.mean(losses_evaluation)}"
print("Validation loss calculated. Saving Validation loss graph...")
f1 = plt.figure(1)
plt.plot(losses_evaluation[::100], label=graph_label)
plt.title('Validation loss Vs. Sample')
plt.xlabel('Every 100th sample')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.savefig(
    "/home/SharedStorage2/NewUsersDir/aledhari/npate181/MachineVisionProject/Final_submission/Predictions/validation_loss.png")
print("Graph saved. Generating predictions...")
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

# ==== EVAL LOOP
model.eval()
torch.set_grad_enabled(False)

# store information for evaluation
future_coords_offsets_pd = []
timestamps = []
confidences_list = []
agent_ids = []
losses_validate = []
progress_bar = tqdm(eval_dataloader)

for data in progress_bar:

    _, preds, confidences = forward(data, model, device)
    # fix for the new environment
    preds = preds.cpu().numpy()
    world_from_agents = data["world_from_agent"].numpy()
    centroids = data["centroid"].numpy()
    coords_offset = []

    # convert into world coordinates and compute offsets
    for idx in range(len(preds)):
        for mode in range(3):
            preds[idx, mode, :, :] = transform_points(preds[idx, mode, :, :], world_from_agents[idx]) - centroids[idx][
                                                                                                        :2]

    future_coords_offsets_pd.append(preds.copy())
    confidences_list.append(confidences.cpu().numpy().copy())
    timestamps.append(data["timestamp"].numpy().copy())
    agent_ids.append(data["track_id"].numpy().copy())

print("Predictions generated. Saving predictions...")
pred_path = 'submission.csv'
write_pred_csv(pred_path,
               timestamps=np.concatenate(timestamps),
               track_ids=np.concatenate(agent_ids),
               coords=np.concatenate(future_coords_offsets_pd),
               confs=np.concatenate(confidences_list)
               )

metrics = compute_metrics_csv(eval_gt_path, pred_path, [neg_multi_log_likelihood, time_displace])
for metric_name, metric_mean in metrics.items():
    print(metric_name, metric_mean)

print("Precitions saved. Generating images...")
model.eval()
torch.set_grad_enabled(False)

# build a dict to retrieve future trajectories from GT
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
    f2 = plt2.figure(v_index)
    plt2.imshow(im_ego[::-1])
    plt2.savefig(
        "/home/SharedStorage2/NewUsersDir/aledhari/npate181/MachineVisionProject/Final_submission/Predictions/final_prediction" + str(
            frame_number)
        + ".png", format='png')

print("Sucessful Run")

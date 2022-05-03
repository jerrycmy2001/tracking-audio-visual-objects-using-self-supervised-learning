import torch
import numpy as np
from tqdm import tqdm
import cv2
import torch.nn.functional as F


def calc_attention_score(video, attention_map):
    '''
    params:
    video:         1 x 3 x T x H x W
    attention_map: N x T x h x w
    '''
    attention_map_with_channels = attention_map.detach().cpu().numpy()
    _, C, T, H, W = video.shape
    N, _, h, w = attention_map.shape
    score_list = []
    for n in tqdm(range(N)):
        tracked_coordinates = np.stack(
            [np.linspace(-1, 1, H, dtype=np.float32)[:, None].repeat(W, axis=1),
             np.linspace(-1, 1, W, dtype=np.float32)[None, :].repeat(H, axis=0)],
            axis=2)  # H x W x 2
        score = np.zeros((H, W))
        last_frame = video[0, :, 0, :, :].detach().cpu().numpy().transpose(1, 2, 0)
        last_frame = cv2.cvtColor(last_frame, cv2.COLOR_RGB2GRAY)
        for t in range(1, T):
            next_frame = video[0, :, t, :, :].detach().cpu().numpy().transpose(1, 2, 0)
            next_frame = cv2.cvtColor(next_frame, cv2.COLOR_RGB2GRAY)
            optical_flow_frame = cv2.calcOpticalFlowFarneback(
                last_frame, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)  # H x W x 2
            optical_flow_coordinates = F.grid_sample(
                input=torch.from_numpy(optical_flow_frame[None, :, :, :].transpose(0, 3, 1, 2)),
                grid=torch.from_numpy(tracked_coordinates[None, :, :, :]))
            optical_flow_coordinates[:, :, 0] /= (H / 2)
            optical_flow_coordinates[:, :, 1] /= (W / 2)
            tracked_coordinates += optical_flow_coordinates.numpy().squeeze().transpose(1, 2, 0)
            score_per_frame = F.grid_sample(
                input=torch.from_numpy(attention_map_with_channels[n, t, :, :][None, None, :, :]),
                grid=torch.from_numpy(tracked_coordinates[None, :, :, :]))
            score += score_per_frame.numpy().squeeze()
            last_frame = next_frame
        score_list.append(score / (T - 1))
    return np.stack(score_list, axis=0)


def track_point(video, points):
    '''
    params:
    video:         1 x 3 x T x H x W
    points:        P x 2

    returns:
    points_trajectory: P x T x 2
    '''
    _, C, T, H, W = video.shape
    last_frame = video[0, :, 0, :, :].detach().cpu().numpy().transpose(1, 2, 0)
    last_frame = cv2.cvtColor(last_frame, cv2.COLOR_RGB2GRAY).astype(np.uint8)
    points_trajectory_list = [points]
    for t in range(1, T):
        next_frame = video[0, :, t, :, :].detach().cpu().numpy().transpose(1, 2, 0)
        next_frame = cv2.cvtColor(next_frame, cv2.COLOR_RGB2GRAY).astype(np.uint8)
        points, _, _ = cv2.calcOpticalFlowPyrLK(last_frame, next_frame, points, None, winSize=(15, 15),
                                                maxLevel=2,
                                                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        points_trajectory_list.append(points)
        last_frame = next_frame
    points_trajectory = np.stack(points_trajectory_list, axis=0)
    return points_trajectory

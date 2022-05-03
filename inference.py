from dataset import VideoDataset
from optical_flow import calc_attention_score, track_point
from net import Net
import torch
from torch.utils.data import DataLoader
from utils import wav2filterbanks
import os
import cv2
import numpy as np

model_path = "./model.pt"
device = "cuda:0"

if __name__ == "__main__":
    model = Net(device=device)
    if not os.path.exists(model_path):
        print("No pretrained model")
        exit(1)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    dataset = VideoDataset(data_path="./inference", resize=(224, 224), cycle=False)
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=0)

    for idx, (video_name, video, audio) in enumerate(dataloader):
        mel = wav2filterbanks(audio)
        mel = mel[:, None, :, :]
        video = video.to(device=device)
        mel = mel.to(device=device)

        attention_map = model(video, mel)

        attention_score = calc_attention_score(video, attention_map)
        attention_score = attention_score[0]
        points = np.array(
            np.unravel_index(np.argmax(attention_score),
                             attention_score.shape),
            dtype=np.float32)[
            None, :]
        points_trajectory = track_point(video, points)

        T = video.shape[2]
        output_path = os.path.join("./inference", "output", video_name[0] + ".avi")
        print(output_path)
        video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MJPG'), 25, (224, 224))
        for t in range(T):
            frame = video[0, :, t, :, :].detach().cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
            length = 10
            frame = cv2.rectangle(
                img=frame,
                pt1=(max(int(points_trajectory[t, 0, 0]) - length, 0),
                     max(int(points_trajectory[t, 0, 1]) - length, 0)),
                pt2=(min(int(points_trajectory[t, 0, 0]) + length, frame.shape[0]),
                     min(int(points_trajectory[t, 0, 1]) + length, frame.shape[1])),
                color=(255, 0, 0),
                thickness=2)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)
        video_writer.release()

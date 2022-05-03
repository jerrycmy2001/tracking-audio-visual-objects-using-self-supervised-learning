import torch
from utils import wav2filterbanks, calculate_loss_max, calculate_loss_mean, sample_video_aligned
import matplotlib.pyplot as plt
import numpy as np

opts = {
    'sample_rate': 16000,
    'fps': 25,
    'input_video_frames': 125,
    'n_samples': 6,
    'model_path': "./model.pt",
    'loss_path': "./loss.png",
    'change_loss_calculation': 500
}


def train(model, dataloader, optimizer, training_loss_list, val_loss_list, num_epoch=10000, device="cpu"):
    factor = opts['sample_rate'] // opts['fps']
    for idx in range(num_epoch):
        optimizer.zero_grad()
        print(("Epoch: {}/{}").format(idx, num_epoch))

        # sample from different sources
        sampled_video_list, sampled_audio_list = [], []
        while len(sampled_audio_list) < opts['n_samples']:
            _, video, audio = next(iter(dataloader))  # support infinite loop
            if len(video) == 0:
                print("Invalid video")
                continue
            sampled_video, sampled_audio = sample_video_aligned(video, audio, opts['input_video_frames'], factor)
            if sampled_video is None:
                print("Invalid video")
                continue
            if len(sampled_video_list) == 0:
                sampled_video_list.append(sampled_video)
            sampled_audio_list.append(sampled_audio)
        assert(len(sampled_audio_list) == opts['n_samples'] and len(sampled_video_list) == 1)
        video = torch.stack(sampled_video_list, dim=0)
        audio = torch.stack(sampled_audio_list, dim=0)
        mel = wav2filterbanks(audio)
        mel = mel[:, None, :, :]
        video = video.to(device=device)
        mel = mel.to(device=device)

        # forward
        attention_map = model(video, mel)

        # backward
        if idx < opts['change_loss_calculation']:
            val_loss = calculate_loss_mean(attention_map)
        else:
            val_loss = calculate_loss_max(attention_map)
        val_loss_list.append(val_loss.item())
        val_loss.backward()
        optimizer.step()

        # training loss
        with torch.no_grad():
            attention_map = model(video, mel)
            if idx < opts['change_loss_calculation']:
                training_loss = calculate_loss_mean(attention_map)
            else:
                training_loss = calculate_loss_max(attention_map)
            training_loss_list.append(training_loss.item())

        print("val loss:", val_loss.item(), "training loss:", training_loss.item())

        # save model
        if idx % 10 == 0:
            print("Saving model...")
            to_save = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'training_loss': training_loss_list,
                'val_loss': val_loss_list}
            torch.save(to_save, opts['model_path'])
        plt.plot(range(len(training_loss_list)), training_loss_list, 'b', label="training loss")
        plt.plot(range(len(val_loss_list)), val_loss_list, 'r', label="val loss")
        plt.savefig(opts['loss_path'])

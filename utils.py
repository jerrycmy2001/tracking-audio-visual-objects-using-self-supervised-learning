import librosa
import torch
import numpy as np
import random
from torch import nn
import torch.nn.functional as F
import torchaudio

audio_opts = {
    'sample_rate': 16000,
    'n_fft': 512,
    'win_length': 320,
    'hop_length': 160,
    'n_mel': 80,
}


def wav2filterbanks(wav, mel_basis=None):
    """
    :param wav: Tensor b x T
    """
    transform = torchaudio.transforms.MelSpectrogram(n_fft=audio_opts['n_fft'],
                                                     hop_length=audio_opts['hop_length'],
                                                     win_length=audio_opts['win_length'],
                                                     #    window_fn=torch.hann_window(audio_opts['win_length']),
                                                     center=True,
                                                     pad_mode='reflect',
                                                     normalized=False,
                                                     onesided=True,
                                                     n_mels=audio_opts['n_mel'],
                                                     f_min=0,
                                                     f_max=int(audio_opts['sample_rate'] / 2))
    mel_list = []
    for w in wav:
        mel_list.append(transform(w).permute(1, 0))
    mel = torch.stack(mel_list, dim=0)

    # normalize
    mean = mel.mean(dim=1, keepdim=True)
    std = mel.std(dim=1, keepdim=True)
    mel = (mel - mean) / std
    return mel


def truncate_video(video, audio, max_length, factor):
    length = video.shape[2]
    if length < max_length:
        # print("Length:", length)
        return None, None
    n_chunks = (length - max_length // 2) // max_length + 1
    video_list, audio_list = [], []
    for n in range(n_chunks):
        video_list.append(video[0, :, n * max_length // 2:n * max_length // 2 + max_length, :, :])
        audio_list.append(audio[0, n * max_length // 2 * factor:(n * max_length // 2 + max_length) * factor])
    video = torch.stack(video_list, dim=0)
    audio = torch.stack(audio_list, dim=0)
    return video, audio


def sample_video_aligned(video, audio, input_video_frames, factor):
    '''
    sample an aligned video and audio from one source
    '''
    video_frames = video.shape[2]
    if video_frames < input_video_frames:
        print("Not enough frames:", video_frames)
        return None, None
    start = random.randint(0, video_frames - input_video_frames)
    sampled_video = video[0, :, start:start + input_video_frames, :, :]
    sampled_audio = audio[0, start * factor:(start + input_video_frames) * factor]
    return sampled_video, sampled_audio


def calculate_loss_max(attention_map):
    '''
    params:
    attention_map: N x T x h x w
    '''
    max_score = attention_map.amax(dim=(2, 3)).mean(dim=1)
    # loss = F.softmax(max_score)[0]
    loss = nn.CrossEntropyLoss()(max_score[None, :], torch.tensor([0], dtype=torch.long, device=max_score.device))
    return loss


def calculate_loss_mean(attention_map):
    '''
    params:
    attention_map: N x T x h x w
    '''
    mean_score = attention_map.mean(dim=(1, 2, 3))
    # loss = F.softmax(mean_score)[0]
    loss = nn.CrossEntropyLoss()(mean_score[None, :], torch.tensor([0], dtype=torch.long, device=mean_score.device))
    return loss

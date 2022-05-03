import os
import av
import numpy as np
from torch.utils import data
from scipy.io import wavfile
import torch
import random


class VideoDataset(data.Dataset):
    def __init__(self, data_path, resize, cycle, fps=25, sample_rate=16000):
        self.resize = resize
        self.fps = fps
        self.sample_rate = sample_rate
        self.data_path = data_path
        data_path_files = os.listdir(data_path)
        self.all_vids = []
        self.cycle = cycle
        for file in data_path_files:
            if file.endswith(".mp4"):
                self.all_vids.append(file)

    def __len__(self):
        if self.cycle:
            return 1
        else:
            return len(self.all_vids)

    def __getitem__(self, index):
        if self.cycle:
            index = random.randint(0, len(self.all_vids) - 1)

        # find video and audio path
        video_path = self.all_vids[index]
        video_name, _ = os.path.splitext(video_path)
        video_path_orig = os.path.join(self.data_path, video_path)
        folder_path_25fps = os.path.join(self.data_path, "25fps")
        if not os.path.exists(folder_path_25fps):
            os.makedirs(folder_path_25fps)
        video_path_25fps = os.path.join(folder_path_25fps, video_name + "_25fps.mp4")
        audio_path = os.path.join(folder_path_25fps, video_name + ".wav")

        # reencode video to 25 fps
        if not os.path.exists(video_path_25fps):
            command = ("ffmpeg -threads 1 -loglevel error -y -i {} -an -r 25 {}".format(video_path_orig, video_path_25fps))
            from subprocess import call
            cmd = command.split(' ')
            print('Resampling {} to 25 fps'.format(video_path_orig))
            return_value = call(cmd)
            if return_value != 0:
                return "", [], []

        # load video
        frame_list = av.open(video_path_25fps)
        images = [frame.to_image() for frame in frame_list.decode(video=0)]
        video_output = np.array([np.array(im) for im in images])
        if self.resize:
            import torchvision
            from PIL import Image
            ims = [Image.fromarray(frm) for frm in video_output]
            ims = [torchvision.transforms.functional.resize(im, self.resize) for im in ims]
            video_output = np.array([np.array(im) for im in ims])

        # load audio
        if not os.path.exists(audio_path):
            command = (
                ("ffmpeg -threads 1 -loglevel error -y -i {} "
                 "-async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 {}").format(video_path_orig, audio_path))
            from subprocess import call
            cmd = command.split(' ')
            return_value = call(cmd)
            if return_value != 0:
                return "", [], []
        _, wav = wavfile.read(audio_path)
        fr_aud = int(np.round(0 * self.sample_rate))
        to_aud = int(np.round(10000 * self.sample_rate))
        wav_output = wav[fr_aud:to_aud].astype('float32')
        # print(wav_output.shape)

        # truncate
        trunkated_video_length = min(
            video_output.shape[0],
            wav_output.shape[0] // self.sample_rate * self.fps) // self.fps * self.fps
        video_output = video_output[:trunkated_video_length]
        wav_output = wav_output[:trunkated_video_length // self.fps * self.sample_rate]

        video_output = torch.from_numpy(video_output).to(torch.float32).permute(3, 0, 1, 2)

        return video_name, video_output, wav_output

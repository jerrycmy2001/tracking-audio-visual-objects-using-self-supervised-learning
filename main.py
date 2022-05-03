import os
import numpy as np
from torch.utils import data
from net import Net
from dataset import VideoDataset
import torch
from torch import optim
from torch.utils.data import DataLoader
from train import train

model_path = "model.pt"

if __name__ == "__main__":
    device = "cuda:0"
    model = Net(device=device)
    val_loss_list, training_loss_list = [], []
    optimizer = optim.Adam(params=model.parameters(), lr=2e-4)

    # load pretrained model
    if os.path.exists(model_path):
        print("Pretrained model found, would you like to use it? y/n")
        print(">> ", end="")
        rep = str(input())
        if rep == "y":
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            val_loss_list = checkpoint['val_loss']
            training_loss_list = checkpoint['training_loss']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    dataset = VideoDataset(data_path="./AVSpeechDownloader/train", resize=(224, 224), cycle=True)
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=True, num_workers=0)
    train(model=model, dataloader=dataloader, optimizer=optimizer,
          training_loss_list=training_loss_list, val_loss_list=val_loss_list, device=device)

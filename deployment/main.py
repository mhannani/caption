# For testing purposes
import torch

from utils.torchscript import torchscript_model

if __name__ == "__main__":
    chkpts_path = "checkpoints_torchscripted/checkpoint_num_39__21_11_2021__16_33_06.pth.tar"
    torchscript_model(chkpts_path)

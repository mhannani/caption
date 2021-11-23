import torch
from utils import load_model


def torchscript_model(chkpts_path):
    """
    Torchscripts saves the model for optimization with Just-In-Time compiler.
    :param chkpts_path: string
        Path to checkpoints
    :return: torchscripted model
    """
    # load the model
    model = load_model(eval_mode=False, checkpoints_path=chkpts_path)

    # torchscript model
    torchscripted_model = torch.jit.script(model)

    # save the model
    torchscripted_model.save("../checkpoints_torchscripted/model.pt")

    return torchscripted_model

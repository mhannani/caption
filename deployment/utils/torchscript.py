import torch


def torchscript_model(model):
    """
    Torchscripts saves the model for optimization with Just-In-Time compiler.
    :param model: pretrained model
    :return: torchscripted model
    """

    # torchscript model
    torchscripted_model = torch.jit.script(model)

    # save the model
    torchscripted_model.save("../checkpoints_deployment/model.pt")

    return torchscripted_model

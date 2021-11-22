import torch
from utils.inference import inference
from utils.data_loader import data_loader
from utils.checkpoints import load_checkpoint
from utils.models import Captioner
import torchvision.transforms as transforms


if __name__ == "__main__":

    # define the transformation
    transform = transforms.Compose(
        [
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    # load the pretrained model
    # hyperparameters
    embed_size = 256
    hidden_size = 256
    vocabulary_size = 2339
    num_layer = 1
    lr = 3e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    model = Captioner(embed_size, hidden_size, vocabulary_size, num_layer).to(device)

    model, _ = load_checkpoint(torch.load("utils/checkpoints/checkpoint_num_39__21_11_2021__16_33_06.pth.tar",
                                          map_location=torch.device('cpu')), model)

    # set up the model to evaluation mode
    model.eval()

    # get the data
    training_data, train_dataset = data_loader(root_dir="Data/Images/train",
                                               caption_file="Data/caption_train.csv",
                                               transform=transform, num_workers=6)

    inference(model, train_dataset, transform, device, image_name="335588286_f67ed8c9f9")

import torch
from utils.data_loader import data_loader
from utils.models import Captioner
from torchvision.transforms import transforms

# Apply some transformation to our data
transform = transforms.Compose(
    [
        transforms.Resize((356, 356)),
        transforms.RandomCrop((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)

# get the data
data, dataset = data_loader(root_dir="Data/Images/train",
                            caption_file="Data/caption_train.csv",
                            transform=transform, num_workers=6)

# hyperparameters
embed_size = 256
hidden_size = 256
vocabulary_size = len(dataset.vocabulary)
num_layer = 1


images, captions = next(iter(data))

model = Captioner(embed_size, hidden_size, vocabulary_size, num_layer)

model.eval()

traced_model = torch.jit.trace(model, (images, captions))
traced_model.save("deployment/checkpoints/caption.pt")






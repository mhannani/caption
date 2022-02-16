import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from utils.data_loader import data_loader
from utils.models import Captioner
from utils.checkpoints import load_checkpoint, save_checkpoint
# from utils.plot import show_image


def train():
    """
    Train the captioner
    :return:
    """

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
    training_data, train_dataset = data_loader(root_dir="Data/Images/train",
                                               caption_file="Data/caption_train.csv",
                                               transform=transform, num_workers=6)

    # get the test data
    test_data, test_dataset = data_loader(root_dir="Data/Images/test",
                                          caption_file="Data/caption_test.csv",
                                          transform=transform, num_workers=6)

    # get validation data
    valid_data, valid_dataset = data_loader(root_dir="Data/Images/valid",
                                            caption_file="Data/caption_valid.csv",
                                            transform=transform, num_workers=6)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_model = False
    save_model = True

    # hyperparameters
    embed_size = 256
    hidden_size = 256
    vocabulary_size = len(train_dataset.vocabulary)
    print(vocabulary_size)
    num_layer = 1
    lr = 3e-4
    num_epochs = 101

    # Tensorboard
    writer = SummaryWriter("runs/flickr")
    step = 0

    # Initialize model
    model = Captioner(embed_size, hidden_size, vocabulary_size, num_layer).to(device)

    # loss function
    criterion = nn.CrossEntropyLoss(ignore_index=train_dataset.vocabulary.stoi["<PAD>"])

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # load checkpoint if already saved
    if load_model:
        _, step = load_checkpoint(torch.load("checkpoint.pth.tar"), model)

    model.train()

    # training process
    for epoch in range(num_epochs):
        print(f'epoch {epoch}/{num_epochs} ... Training loss: -')

        running_loss = 0.0

        # Go through batches 
        for index, (images, captions) in enumerate(training_data):
            images = images.to(device)
            captions = captions.to(device)
            output = model(images, captions[:-1])

            loss = criterion(output.reshape(-1, output.shape[2]), captions.reshape(-1))

            writer.add_scalar("Training loss", loss.item(), global_step=step)
            step += 1
            optimizer.zero_grad()
            loss.backward(loss)
            optimizer.step()

            # print statistics
            # accumulate the training loss
            running_loss += loss.item()

            print(f"current batch {index + 1} / {len(training_data)}")

        # print(f'Epoch: {epoch + 1}/{num_epochs} ... Training loss: {running_loss / len(training_data)}')

        # save the model at this stage
        if save_model:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step
            }
        # save model each 10 epochs
            if (epoch + 1) % 10 == 0:
                save_checkpoint(checkpoint, epoch)

                # generate the caption
                # model.eval()
                # with torch.no_grad():
                #     dataiter = iter(valid_data)
                #     img, _ = next(dataiter)
                #     features = model.encoder(img[0:1].to(device))
                #     caps = model.decoder.generate_caption(features.unsqueeze(0), vocab=train_dataset.vocab)
                #     caption = ' '.join(caps)
                #     show_image(img[0], title=caption)

                # model.train()
    print('Finished Training')


if __name__ == "__main__":
    train()


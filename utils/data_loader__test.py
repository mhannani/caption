from data_loader import data_loader

data_loader__test = data_loader(root_dir="../Data/Images/",
                                caption_file="../Data/captions.txt",
                                transform=None)

if __name__ == '__main__':

    for i, (image, caption) in enumerate(data_loader__test):
        print(image.shape)
        print(caption.shape)

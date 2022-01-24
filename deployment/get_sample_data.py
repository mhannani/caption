

# get the data
training_data, train_dataset = data_loader(root_dir="../Data/Images/train",
                                           caption_file="../Data/caption_train.csv",
                                           transform=transform, num_workers=6)
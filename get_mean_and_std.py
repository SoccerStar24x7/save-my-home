import torch

# determines the mean based on the given dataset, which in our case are the photos of the stove
def get_mean_std(train_loader):

    batch_mean = torch.zeros(3)
    batch_mean_sqrd = torch.zeros(3)

    for batch_data, _ in train_loader:
        batch_mean += batch_data.mean(dim=(0, 2, 3))  # E[batch_i]
        batch_mean_sqrd += (batch_data**2).mean(dim=(0, 2, 3))  #  E[batch_i**2]

    mean = batch_mean / len(train_loader) # calculates the mean

    var = (batch_mean_sqrd / len(train_loader)) - (mean**2)

    std = var**0.5

    return mean, std # returns the mean and the std of the given dataset
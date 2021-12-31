# EECS 545 Fall 2021
import math
# from typing_extensions import TypeVarTuple
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    """
    Convolutional Neural Network.
    """
    def __init__(self):
        super().__init__()

        # TODO (part c): define layers
        self.conv1 = nn.Conv2d(3, 16, 5, stride=2, padding=2)  # convolutional layer 1
        self.conv2 = nn.Conv2d(16, 32, 5, stride=2, padding=2) # convolutional layer 2
        self.conv3 = nn.Conv2d(32, 64, 5, stride=2, padding=2) # convolutional layer 3
        self.conv4 = nn.Conv2d(64, 128, 5, stride=2, padding=2) # convolutional layer 4
        self.fc1 = nn.Linear(128*2*2, 64, bias=True) # fully connected layer 1
        self.fc2 = nn.Linear(64, 5, bias=True) # fully connected layer 2 (output layer)

        self.init_weights()

    def init_weights(self):
        for conv in [self.conv1, self.conv2, self.conv3, self.conv4]:
            C_in = conv.weight.size(1)
            nn.init.normal_(conv.weight, 0.0, 1/math.sqrt(5 * 2.5 * C_in))
            nn.init.constant_(conv.bias, 0.0)

        # TODO (part c): initialize parameters for fully connected layers
        nn.init.normal_(self.fc1.weight, 0.0, 1/math.sqrt(256))
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.normal_(self.fc2.weight, 0.0, 1/math.sqrt(32))
        nn.init.constant_(self.fc2.bias, 0.0)

    def forward(self, x):
        N, C, H, W = x.shape

        # TODO (part c): forward pass of image through the network
        z = F.relu(self.conv1(x))
        z = F.relu(self.conv2(z))
        z = F.relu(self.conv3(z))
        z = F.relu(self.conv4(z))
        z = z.view(z.size(0),-1)
        z = F.relu(self.fc1(z))
        z = self.fc2(z)
        return z    


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    from dataset import DogDataset
    net = CNN()
    print(net)
    print('Number of CNN parameters: {}'.format(count_parameters(net)))
    dataset = DogDataset()
    images, labels = iter(dataset.train_loader).next()
    print('Size of model output:', net(images).size())
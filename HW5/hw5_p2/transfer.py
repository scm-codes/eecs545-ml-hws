# EECS 545 Fall 2021
import torch
import torchvision.models as models
from dataset import DogDataset
from train import train


def load_pretrained(num_classes=5):
    """
    Load a ResNet-18 model from `torchvision.models` with pre-trained weights. Freeze all the parameters besides the
    final layer by setting the flag `requires_grad` for each parameter to False. Replace the final fully connected layer
    with another fully connected layer with `num_classes` many output units.
    Inputs:
        - num_classes: int
    Returns:
        - model: PyTorch model
    """
    # TODO (part f): load a pre-trained ResNet-18 model
    resnet18 = models.resnet18(pretrained=True)
    for param in resnet18.parameters():
        param.requires_grad = False
    
    # add a final fully connected layer
    num_ftrs = resnet18.fc.in_features
    resnet18.fc = torch.nn.Linear(num_ftrs, num_classes)
    return resnet18


if __name__ == '__main__':
    config = {
        'dataset_path': 'data/images/dogs',
        'batch_size': 4,
        'if_resize': False,             
        'ckpt_path': 'checkpoints/transfer',
        'plot_name': 'Transfer',
        'num_epoch': 5,
        'learning_rate': 1e-3,
        'momentum': 0.9,
    }
    dataset = DogDataset(config['batch_size'], config['dataset_path'],config['if_resize'])
    model = load_pretrained()
    train(config, dataset, model)
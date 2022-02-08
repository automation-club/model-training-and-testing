from torch import nn
from torchvision import models

import torch

class Resnet(nn.Module):
    
    def __init__(self, device, output_dim):
        super(Resnet, self).__init__()
        self.device = device
        self.model = self.create_model(output_dim)

    def create_model(self, output_dim):
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(512, output_dim)
        return model

    def train_model(self, loader, epochs,):
        optim = torch.optim.Adam(self.model.parameters(), lr=3e-4)
        criterion = torch.nn.CrossEntropyLoss()

        self.model.to(self.device)

        for epoch in range(epochs):
            for idx, (input, target) in enumerate(loader):
                input = input.to(self.device)
                target = target.to(self.device)

                print(input)
                
                output = self.model(input)

                print("calculated")
                loss = criterion(output, target)

                print("reached")
                optim.zero_grad()
                loss.backward()
                optim.step()

                

                if idx % 100 == 0:
                    print(f"Epoch {epoch}, Batch {idx}")
                    print(f"Loss: {loss.item()}")


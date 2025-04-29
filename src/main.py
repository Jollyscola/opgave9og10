import json
import torch
from torch import nn, optim,accelerator,Tensor
import photo 
from typing import Callable
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.nn import Flatten,Sequential,Module,Linear,ReLU



def load_config():
    with open("config.json", "r") as op:
      config_json = json.load(op)
    print(config_json["Epochs"])
    return config_json



traning_ds = datasets.FashionMNIST(
    root='data', 
    train=True, 
    download = True, 
    transform=transforms.ToTensor()
    )


test_ds = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=transforms.ToTensor()
)



config = load_config()

batch_size = config["BatchSize"]
learning_rate = config["LearningRate"]
epochs = config["Epochs"]




class Neural_network(Module):
    def __init__(self): 
        super().__init__() 
        # photo.show_more_image(test_ds)
        self.flatten: Flatten = Flatten() 
        self.network_stack: Sequential = Sequential( 
        Linear(in_features = 28*28, out_features = 512), 
        ReLU(), 
        Linear(in_features = 512, out_features = 10) 
        )
    def forward(self,x): 
        x = self.flatten(x) 
        output = self.network_stack(x) 
        return output
    def __str__(self):
        return f"Neural_network: {self.network_stack}"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Neural_network().to(device)
print(model)



def train_loop(
    dataloader: DataLoader,
    model: Module,
    loss_in: Callable[[Tensor, Tensor], Tensor],
    optimizer,
    batch_size: int,
):
    size = len(dataloader.dataset)
    model.train()
    for batch, (x,y) in enumerate(dataloader):
        pred = model(x)
        loss:Tensor = loss_in(pred,y)

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss = loss.item()
            current = batch * batch_size + len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


train_loader = DataLoader(traning_ds, batch_size=batch_size, shuffle=True)

# Loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Run training
for epoch in range(epochs):
    print(f"Epoch {epoch+1}\n------------------------------")
    train_loop(train_loader, model, loss_fn, optimizer, batch_size)
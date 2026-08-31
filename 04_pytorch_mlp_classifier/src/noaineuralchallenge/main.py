import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

learning_rate = 0.001

torch.manual_seed(42)
X = torch.randn(50, 5, dtype=torch.float32)
Y = torch.randint(0, 4, (50,))

dataset = TensorDataset(X, Y)
dataloader = DataLoader(dataset, batch_size=64)

class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hl = nn.Linear(5, 8)
        self.relu = nn.ReLU()
        self.ol = nn.Linear(8, 4)
    
    def forward(self, x):
        x = self.hl(x)
        x = self.relu(x)
        logits = self.ol(x)
        return logits

model = NeuralNet()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.train()
for epoch in range(10000):
    for inputs, targets in dataloader:
        logits = model(inputs)

        loss = loss_fn(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


model.eval()
with torch.no_grad():
    prediciones = torch.argmax(model(X), dim=1)
    aciertos = (prediciones == Y)
    accuracy = aciertos.float().mean()

    print(accuracy.item() * 100)
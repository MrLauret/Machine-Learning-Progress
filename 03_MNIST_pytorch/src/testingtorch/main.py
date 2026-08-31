import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import os

class RedMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )
    
    def forward(self, x):
        return self.layers(x)

def main() -> None:
    table = pd.read_csv("mnist_test.csv")
    labels = table.iloc[:, 0]
    pixels = table.iloc[:, 1:] / 255.0

    pixels_tensor = torch.tensor(pixels.values, dtype=torch.float32)
    labels_tensor = torch.tensor(labels.values, dtype=torch.long)

    dataset = TensorDataset(pixels_tensor, labels_tensor)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    path_model = "mnist_model.pt"
    model = RedMNIST()

    if not os.path.exists(path_model):
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        model.train()
        for epoch in range(50):
            for batch_pixels, batch_labels in loader:

                outputs = model(batch_pixels)

                loss = loss_fn(outputs, batch_labels)

                optimizer.zero_grad()
                
                loss.backward()

                optimizer.step()

            if epoch % 5 == 0:
                loss_str = format(loss.item(), '.25f')
                print(f"\nEpoch: {epoch}\nLoss: {loss_str}")
        
        torch.save(model.state_dict(), "mnist_model.pt")
        print("Trained model saved")
    else:
        load_state = torch.load(path_model)
        model.load_state_dict(load_state)

        with torch.no_grad():
            model.eval()

            outputs = model(pixels_tensor)
            prediction = outputs.argmax(dim=1)

            right_guesses = (prediction == labels_tensor).sum().item()

            total = len(labels_tensor)
            accuracy = (right_guesses / total) * 100

            print(f"\nTotal: {total}")
            print(f"Right guesses: {right_guesses}")
            print(f"Accuracy: {accuracy}")

if __name__ == "__main__":
    main()
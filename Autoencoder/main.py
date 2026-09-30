import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from PIL import Image
import torchvision
import os
from matplotlib import pyplot as plt
torch.set_printoptions(sci_mode=False)

LR = 0.0001
EPOCHS = 100
BATCH_SIZE = 64
MODEL_PATH = "AutoencoderModel.pth"
DATA_PATH = "Training_Data"
TESTING_PATH = "Testing_Images"

class AutoencoderNetwork(nn.Module):
    def __init__(self):
        super(AutoencoderNetwork, self).__init__()

        """
        Conv2d hace convolución en 2 dimensiones #
        
        3 pasos del proceso convolucional
            * La imagen de entrada: Es una matriz de números, si es RGB son 3 matrizes superpuestas
            * El filtro o Kernel:   Es una matriz normalmente de 3x3 o 5x5 de pesos, estos serán ajustados durante el entreno
                Superposición:          Un filtro de 3x3 va a colocarce en la esquina superior izquierda
                Multiplicación:         El valor del pixel de debajo con el valor del filtro de arriba
                Suma:                   Se suman esos 9 resultados y se colocan en el primer pixel del mapa de caracteristicas
                Desplazamiento:         Mueves el filtro un pixel a la derecha y repites hasta acabar la fila,
                                        cuando acabes la fila bajas un pixel y empiezas en la izquierda otra vez

            * Mapa de caracteristicas: Es la nueva imagen resultante, contiene la información original simplificada
            * ReLU: Si un número es negativo lo convierte a 0, si no lo deja igual
        """

        self.encoder = nn.Sequential(
            
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=True),
            
            nn.ReLU(),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=True),

            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1,bias=True),

            nn.ReLU(),
            
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True),

            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)

    def encode(self, x):
        return self.encoder(x)

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((64, 64)),
    torchvision.transforms.ToTensor()
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoencoderNetwork().to(device=device)
loss = None

if not os.path.exists(MODEL_PATH):

    img_list = []

    for _, _, files in os.walk(f"{DATA_PATH}"):
        for file in files:
            image = Image.open(f"{DATA_PATH}/{file}").convert("RGB")
            flat_tensor = transform(image)
            img_list.append(flat_tensor)

    #torch.stack concatena (añade uno detrás de otro) tensores y devuelve un tensor
    img_tensor = torch.stack(img_list)

    dataset = TensorDataset(img_tensor)
    dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    print("Training...")
    model.train()
    for epoch in range(EPOCHS):

        for i, batch in enumerate(dataloader):
            batch_x = batch[0].to(device)

            logits = model(batch_x)

            loss = criterion(logits, batch_x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 5 == 0:
            print(f"\nEpoch: {epoch}")
            print(f"Loss: {loss.item()}")

    torch.save(model.state_dict(), MODEL_PATH)
else:
    print("Loading model")
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True, map_location=device))

some_number = input("Name of file: ")
img_file_path = os.path.join(TESTING_PATH, f"{some_number}.jpg")

if os.path.exists(img_file_path):
    #.Unsqueeze(): Coge un tensor y le añade una columna de unos
    img_tensor_dev = transform(Image.open(img_file_path).convert("RGB")).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(img_tensor_dev)
        encoded_features = model.encode(img_tensor_dev)

    #.Squeeze():   Coge la columna de unos especificada en los parametros y la borra
    reconstructed_img = logits.squeeze(0).cpu()
    encoded_img = encoded_features.squeeze(0)[0].unsqueeze(0).cpu()

    torchvision.utils.save_image(reconstructed_img, f"Outputs/output{some_number}.jpg")
    torchvision.utils.save_image(encoded_img, f"Embeddeds/encoded{some_number}.jpg")

    print(f"\nDimensión Codificada (Latent): {encoded_features.size()}")
    print(f"Dimensión Entrada: {img_tensor_dev.size()}")
else:
    print(f"El archivo {img_file_path} no existe.")

if loss is not None:
    print(f"\nLoss Final: {loss.item()}")
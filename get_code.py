from vae_gemini_deeper import *
import cv2
import numpy as np
import torch

latent_dim = 100

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VAE(latent_dim).to(device).eval()
state = torch.load("vae_gemini_bigfaces_deeper.pth", map_location=device)
model.load_state_dict(state) # oder state, falls nur Gewichte

img_path = "img.jpg"
img_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

img_f32 = (img_rgb.astype(np.float32) / 255.0)
x = torch.from_numpy(img_f32).permute(2, 0, 1).unsqueeze(0).to(device)   # [1,3,32,32], float32

with torch.no_grad():
    mu, logvar = model.encode(x)
    z = model.reparameterize(mu, logvar)
    arr = z.detach().cpu().numpy().flatten()
    with open("code_img", "w") as f:
        text = ",".join(map(str, arr))
        f.write(text)

    import matplotlib.pyplot as plt

    output = model.decode(z).detach().squeeze().cpu().numpy().transpose(1,2,0)

    import matplotlib.pyplot as plt
    plt.imshow(output)
    plt.show()
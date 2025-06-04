import torch
from scripts.models.R2AU_dynamic import R2AttU_Net
from scripts.preprocessing.preprocessing import preprocess_image  # <-- Correct import
import matplotlib.pyplot as plt
import cv2
import os

# --- MODEL SETUP ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = R2AttU_Net(
    img_ch=1,
    output_ch=1,
    t=4,
    base_filters=16,
    depth=5,
).to(device)

script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "model", "model.pth")
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# --- IMAGE PREPROCESSING ---
folder_path = "./uploaded_image"
file_list = os.listdir(folder_path)
if file_list:
    img_path = os.path.join(folder_path, file_list[0])  # Picks the first file
else:
    raise FileNotFoundError("No files found in the folder.")

img_tensor = preprocess_image(img_path)  # Returns torch tensor [1, 1, H, W]
img_tensor = img_tensor.to(device)

# --- PREDICTION ---
with torch.no_grad():
    output = model(img_tensor)
    prediction = torch.sigmoid(output)

# Convert prediction to numpy and post-process as needed
pred_mask = (prediction.cpu().numpy() > 0.5).astype("uint8")[0, 0]  # shape: [H, W]

# Load original image for display
orig_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(orig_img, cmap="gray")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Predicted Mask")
plt.imshow(pred_mask, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.savefig("prediction_result.png")  # Save the figure to a file
plt.show()  # Optional: will try to display if possible
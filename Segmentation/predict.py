import torch
from scripts.models.R2AU_dynamic import R2AttU_Net
from scripts.preprocessing.preprocessing import preprocess_image  # <-- Correct import
import matplotlib.pyplot as plt
import cv2

# --- MODEL SETUP ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = R2AttU_Net(
    img_ch=1,
    output_ch=1,
    t=4,
    base_filters=16,
    depth=5,
).to(device)
model.load_state_dict(torch.load("/home/ahmd/MLOps/best_last_model/run_20250531_144959_best_model.pth", map_location=device))
model.eval()

# --- IMAGE PREPROCESSING ---
img_path = "data/database/9-1.txt.jpg"
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
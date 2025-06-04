import cv2
import numpy as np

# mask preprocessing
def normalize_mask(mask: np.ndarray) -> np.ndarray:
    mask = (mask > 0.999).astype(np.uint8)


    return mask

# images preprocessing
def uint_to_float(image: np.ndarray) -> np.ndarray:
    return image.astype(np.float32) / 255.0

def resize_array(image: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST)

def reshape_array(image: np.ndarray, height: int, width: int) -> np.ndarray:
    # resize image to shape [C, H, W] or [H, W] --> [1, H, W]
    right_shape = (1, height, width)
    if image.shape == right_shape:
        return image

    resized = resize_array(image, width, height)

    if len(resized.shape) == 2:
        resized = np.expand_dims(resized, axis=0)
    
    if resized.shape == right_shape:
        return resized
    else:
        print(f'Error: the shape of the resized is {resized.shape} and not {right_shape}')
        return None

def preprocess_image(img_path: str, height: int = 256, width: int = 384) -> np.ndarray:
    """
    Loads an image, converts to float, resizes, and reshapes for model input.
    Returns a torch tensor of shape [1, 1, H, W].
    """
    import torch

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = uint_to_float(img)
    img = reshape_array(img, height, width)  # shape: [1, H, W]
    img_tensor = torch.from_numpy(img).unsqueeze(0)  # shape: [1, 1, H, W]
    return img_tensor


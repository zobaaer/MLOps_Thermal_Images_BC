from PIL import Image 
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from skimage.measure import label, regionprops
import flyr


def plot_image_grey_scale(image: np.ndarray):
    image = np.array(image)
    plt.imshow(image, cmap='gray')
    plt.show()

def plot_image_rgb(image: np.ndarray):
    image = np.array(image)
    plt.imshow(image, cmap='jet')
    plt.show()

def load_tiff(file_path: str) -> np.ndarray:
    mask = Image.open(file_path)
    return np.array(mask)

def load_png(file_path: str) -> np.ndarray:
    image = Image.open(file_path)
    return np.array(image)

#resize image to width and height
def resize_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
    image = Image.fromarray(image)
    image = image.resize((width, height))
    return np.array(image)

def crop_breasts(image, mask):
    # Converte la maschera in binaria (0/1)
    bin_mask = (mask > 0).astype(np.uint8)
    # Etichetta le regioni connesse della maschera
    labeled_mask = label(bin_mask)
    props = regionprops(labeled_mask)
    
    # Ordina le regioni in base alla coordinata minc (bounding box più a sinistra)
    props_sorted = sorted(props, key=lambda region: region.bbox[1])
    
    cropped_images = []
    cropped_masks = []
    
    for region in props_sorted:
        minr, minc, maxr, maxc = region.bbox
        cropped_img = image[minr:maxr, minc:maxc]
        cropped_msk = mask[minr:maxr, minc:maxc]
        cropped_images.append(cropped_img)
        cropped_masks.append(cropped_msk)
    
    return cropped_images, cropped_masks

# create a script that merge toghether the masks
def masks_merger(path_data: str):
    masks_path = os.path.join(path_data, 'masks')
    right_mask_path = os.path.join(masks_path, 'MD')
    left_mask_path = os.path.join(masks_path, 'ME')
    preprocessed_masks_path = os.path.join(path_data, 'mask_merged')

    if not os.path.exists(preprocessed_masks_path):
        os.makedirs(preprocessed_masks_path)

    right_masks_files = sorted(os.listdir(right_mask_path))
    left_masks_files = sorted(os.listdir(left_mask_path))

    for right_mask_file, left_mask_file in zip(right_masks_files, left_masks_files):
        id_subject_right = right_mask_file.split('_')[0]
        id_subject_left = left_mask_file.split('_')[0]

        # check that the id of the subject is the same
        if id_subject_right != id_subject_left:
            print(f'Error: {id_subject_right} is different from {id_subject_left}')
            continue

        # check that, in the preprocessed masks folder, there is not already the image
        if id_subject_right + '.npy' in os.listdir(preprocessed_masks_path):
            continue

        # load the images
        right_mask_image = load_tiff(os.path.join(right_mask_path, right_mask_file))
        left_mask_image = load_tiff(os.path.join(left_mask_path, left_mask_file))
        mask_merged = right_mask_image + left_mask_image

        # save the image
        np.save(os.path.join(preprocessed_masks_path, id_subject_right + '.npy'), mask_merged)

import os
import numpy as np
from PIL import Image

def convert_jpg_to_npy(input_dir: str, output_dir: str):
    """
    Converte tutte le immagini .jpg in una cartella in array numpy (.npy)
    e le salva in un'altra cartella mantenendo lo stesso nome (senza estensione .jpg).
    
    Args:
        input_dir (str): path della cartella con immagini .jpg
        output_dir (str): path della cartella dove salvare i .npy
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".jpg"):
            # Path completo all'immagine
            img_path = os.path.join(input_dir, filename)
            img = Image.open(img_path)
            img_np = np.array(img)
            
            output_filename = os.path.splitext(filename)[0] + ".npy"
            output_path = os.path.join(output_dir, output_filename)
            
            # Salva
            np.save(output_path, img_np)
            print(f"Salvato: {output_path}")

def convert_raw_flir_to_thermal(raw_data_path):
    image = flyr.unpack(raw_data_path)
    image = image.celsius
    return np.array(image)

def convert_raw_flir_to_numpy(input_dir: str, output_dir: str):
    """
    Converte tutte le immagini .raw in una cartella in array numpy (.npy)
    e le salva in un'altra cartella mantenendo lo stesso nome (senza estensione .raw).
    
    Args:
        input_dir (str): path della cartella con immagini .raw
        output_dir (str): path della cartella dove salvare i .npy
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".jpg"):
            # Path completo all'immagine
            img_path = os.path.join(input_dir, filename)
            img_np = convert_raw_flir_to_thermal(img_path)
            
            output_filename = os.path.splitext(filename)[0] + ".npy"
            output_path = os.path.join(output_dir, output_filename)
            
            # Salva
            np.save(output_path, img_np)
            print(f"Salvato: {output_path}")

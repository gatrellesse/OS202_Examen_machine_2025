from mpi4py import MPI
from PIL import Image
import numpy as np
from scipy import signal
import os
import time

# Initialisation MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def double_size_block(image_block):
    """Double la taille d'un bloc d'image avec flou et netteté"""
    img = np.array(image_block, dtype=np.double)
    img = np.repeat(np.repeat(img, 2, axis=0), 2, axis=1) / 255.

    # Filtre flou gaussien
    mask_blur = np.array([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]]) / 16.
    blur_image = np.zeros_like(img, dtype=np.double)
    for i in range(3):
        blur_image[:, :, i] = signal.convolve2d(img[:, :, i], mask_blur, mode='same', boundary='symm')

    # Filtre de netteté sur la luminance (V en HSV)
    mask_sharp = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpen_image = np.zeros_like(img, dtype=np.double)
    sharpen_image[:, :, :2] = blur_image[:, :, :2]
    sharpen_image[:, :, 2] = np.clip(signal.convolve2d(blur_image[:, :, 2], mask_sharp, mode='same', boundary='symm'), 0., 1.)

    # Convertir en format image
    sharpen_image = (255. * sharpen_image).astype(np.uint8)
    return Image.fromarray(sharpen_image, 'HSV')

if rank == 0:
    # --- PROCESSUS MAÎTRE ---
    start_time = time.time()
    
    # Charger l'image et convertir en HSV
    path = "datas/"
    image_path = path + "paysageResized.jpg"
    img = Image.open(image_path).convert('HSV')
    img = np.array(img)

    # Diviser l'image en blocs pour les esclaves
    h, w, _ = img.shape
    num_blocks = size - 1  # Nombre d'esclaves disponibles
    block_height = h // num_blocks

    # Envoi des blocs aux esclaves
    for i in range(1, size):
        start_row = (i - 1) * block_height
        end_row = start_row + block_height if i < num_blocks else h
        comm.send(img[start_row:end_row, :, :], dest=i, tag=i)

    # Réception des blocs traités
    new_blocks = []
    for i in range(1, size):
        new_blocks.append(comm.recv(source=i, tag=i))

    # Reconstruction de l'image finale
    final_image = np.vstack(new_blocks)
    final_image = Image.fromarray(final_image, 'HSV').convert('RGB')

    # Sauvegarde de l'image
    output_path = "sorties/paysage_double.jpg"
    final_image.save(output_path)

    end_time = time.time()
    print(f"Image sauvegardée : {output_path}")
    print(f"Temps d'exécution : {end_time - start_time:.2f} secondes")

else:
    # --- PROCESSUS ESCLAVE ---
    received_block = comm.recv(source=0, tag=rank)
    processed_block = double_size_block(received_block)
    comm.send(np.array(processed_block), dest=0, tag=rank)

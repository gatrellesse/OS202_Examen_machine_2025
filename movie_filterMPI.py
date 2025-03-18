# Ce programme va charger n images et y appliquer un filtre de netteté
# puis les sauvegarder dans un dossier de sortie

from PIL import Image
import os
import numpy as np
from scipy import signal
from mpi4py import MPI
import time

# MPI initialisation
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Fonction pour appliquer un filtre de netteté à une image
def apply_filter(image):
    # On charge l'image
    img = Image.open(image)
    print(f"Taille originale {img.size}")
    # Conversion en HSV :
    img = img.convert('HSV')
    # On convertit l'image en tableau numpy et on normalise
    img = np.repeat(np.repeat(np.array(img), 2, axis=0), 2, axis=1)
    img = np.array(img, dtype=np.double)/255.
    print(f"Nouvelle taille : {img.shape}")
    # Tout d'abord, on crée un masque de flou gaussien
    mask = np.array([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]]) / 16.
    # On applique le filtre de flou
    blur_image = np.zeros_like(img, dtype=np.double)
    for i in range(3):
        blur_image[:,:,i] = signal.convolve2d(img[:,:,i], mask, mode='same')
    # On crée un masque de netteté
    mask = np.array([[0., -1., 0.], [-1., 5., -1.], [0., -1., 0.]])
    # On applique le filtre de netteté
    sharpen_image = np.zeros_like(img)
    sharpen_image[:,:,:2] = blur_image[:,:,:2]
    sharpen_image[:,:,2] = np.clip(signal.convolve2d(blur_image[:,:,2], mask, mode='same'), 0., 1.)

    sharpen_image *= 255.
    sharpen_image = sharpen_image.astype(np.uint8)
    # On retourne l'image modifiée
    return Image.fromarray(sharpen_image, 'HSV').convert('RGB')


path = "datas/perroquets/"
# On crée un dossier de sortie
if not os.path.exists("sorties/perroquets"):
    os.makedirs("sorties/perroquets")
out_path = "sorties/perroquets/"

output_images = []

# Nombre total d'images
num_images = 37

images_per_process = num_images // size
start = rank * images_per_process
end = num_images if rank == size - 1 else start + images_per_process


# Distribuer les images aux processus
images_per_process = num_images // size
remainder = num_images % size # Nombre d'images restantes

# Chaque processus traite une partie des images
start_index = rank * images_per_process + min(rank, remainder)
end_index = start_index + images_per_process + (1 if rank < remainder else 0)

# Liste pour stocker les images traitées
output_images = []
start = time.time()
# Traitement des images attribuées à ce processus
for i in range(start_index, end_index):
    image = path + "Perroquet{:04d}.jpg".format(i + 1)
    sharpen_image = apply_filter(image)
    output_images.append((i, sharpen_image))
    print(f"Process {rank}: Image {i + 1} traitée")

# Collecter les résultats sur le processus maître
gathered_output = comm.gather(output_images, root=0)

# Sauvegarder les images sur le processus maître
if rank == 0:
    # Fusionner les résultats de tous les processus
    all_output_images = []
    for output in gathered_output:
        all_output_images.extend(output)
    # Trier les images par index
    all_output_images.sort(key=lambda x: x[0])
    # Sauvegarder les images
    for idx, img in all_output_images:
        img.save(out_path + "Perroquet{:04d}.jpg".format(idx + 1))
    print("Toutes les images ont été sauvegardées.")
    end = time.time()
    print("Time used: ", end - start)
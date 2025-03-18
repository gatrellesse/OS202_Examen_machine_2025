from PIL import Image
import numpy as np
from scipy import signal
from mpi4py import MPI
import time

def double_size(image):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nbp = comm.Get_size()

    if rank == 0:
        img = Image.open(image)
        print(f"Taille originale {img.size}")
        img = img.convert('HSV')
        img_array = np.array(img, dtype=np.float64) / 255.0
        doubled_img = np.repeat(np.repeat(img_array, 2, axis=0), 2, axis=1)
        height, width, _ = doubled_img.shape
        print(f"Nouvelle taille : {doubled_img.shape}")
    else:
        height, width = None, None
        doubled_img = None

    # Broadcast dimensions
    height = comm.bcast(height, root=0)
    width = comm.bcast(width, root=0)

    # Diviser l'image en segments pour chaque processus
    rows_per_proc = height // nbp
    extra_rows = height % nbp
    counts = [(rows_per_proc + 1) * width * 3 if i < extra_rows else rows_per_proc * width * 3 for i in range(nbp)]
    displacements = [sum(counts[:i]) for i in range(nbp)]

    # Allocation de la mémoire pour le sous-image local
    local_size = counts[rank] // 3  # Nombre de pixels (3 canaux)
    local_img = np.zeros((local_size // width, width, 3), dtype=np.float64)

    if rank == 0:
        global_img = doubled_img.flatten()
    else:
        global_img = None

    # Scatterv pour distribuer les morceaux d'image
    comm.Scatterv([global_img, counts, displacements, MPI.DOUBLE], local_img.flatten(), root=0)

    print(f"Process {rank} received chunk {local_img.shape}")
    
    ##### TRAITEMENT LOCAL #####
    # Appliquer les filtres (Flou + Netteté)
    mask_blur = np.array([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]]) / 16.
    mask_sharpen = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

    blur_image = np.zeros_like(local_img)
    for i in range(3):
        blur_image[:,:,i] = signal.convolve2d(local_img[:,:,i], mask_blur, mode='same', boundary='symm')

    sharpen_image = np.zeros_like(local_img)
    sharpen_image[:,:,:2] = blur_image[:,:,:2]
    sharpen_image[:,:,2] = np.clip(signal.convolve2d(blur_image[:,:,2], mask_sharpen, mode='same', boundary='symm'), 0., 1.)

    sharpen_image = (255. * sharpen_image).astype(np.uint8)
    ##### FOM DU TRAITEMENT LOCAL #####
    
    # Gatherv pour reconstruire l'image complète
    if rank == 0:
        final_image = np.zeros((height, width, 3), dtype=np.uint8)
    else:
        final_image = None

    comm.Gatherv(sharpen_image.flatten(), [final_image, counts, displacements, MPI.UNSIGNED_CHAR], root=0)

    if rank == 0:
        final_image = Image.fromarray(final_image, 'HSV').convert('RGB')
        return final_image

# Exécution principale
path = "datas/"
image = path + "paysageResized.jpg"
start_time = time.time()
doubled_image = double_size(image)
end_time = time.time()

if MPI.COMM_WORLD.Get_rank() == 0 and doubled_image:
    doubled_image.save("sorties/paysage_double.jpg")
    print(f"Image sauvegardée en {end_time - start_time:.2f} secondes.")

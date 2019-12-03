
from torchvision import transforms,datasets

from skimage.measure import compare_ssim

import cv2

class ImageFolderWithPaths(datasets.ImageFolder):

    # modifico solo la path
    def __getitem__(self, index):
        # recupero out precedente
        # ritorna una tupla con immagine,categoria
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)

        # print (original_tuple)
        # path  immagine
        path = self.imgs[index][0]


        return original_tuple[0], original_tuple[1], path

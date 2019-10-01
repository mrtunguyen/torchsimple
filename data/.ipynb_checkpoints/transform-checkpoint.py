from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
import cv2
from PIL import Image, ImageOps
import random
import collections
import numpy as np
import PIL
from PIL import Image
import torch
from torchvision.transforms import Normalize
from typing import Callable, Dict, Tuple


IMAGENET_STATS = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
NOMALIZE_STATS = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

class Transformer:
    def __init__(self, key: str, transform_fn: Callable) -> None:
        self.key = key
        self.transform_fn = transform_fn

    def __call__(self,
                 datum: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        datum = datum
        datum[self.key] = self.transform_fn(datum[self.key])
        return datum


def to_torch(scale_factor: float = 255.) -> Callable:
    return lambda x: torch.from_numpy(
        x.astype(np.float32) / scale_factor).permute(2, 0, 1)


def normalize(stats: Tuple = NOMALIZE_STATS) -> Normalize:
    return Normalize(*stats)


class ResizeFixedSize(object):
    """Resize image with fixed width and fixed height
    """
    def __init__(self, height, width):
        self.height = height
        self.width = width
    
    def __call__(self, image_array):
        img = Image.fromarray(image_array)
        hpercent = self.height / float(img.size[1])
        wsize = int((float(img.size[0])*float(hpercent)))
        wsize = 2*int(wsize/2)

        img = img.resize((wsize,self.height), Image.ANTIALIAS)
        if wsize< self.width:
            img = ImageOps.expand(img,border=(int((self.width-wsize)/2),0),fill='white')
        else: 
            img = img.resize((self.width,self.height), Image.ANTIALIAS)

        img = np.array(img, dtype = np.float64)
        img = np.expand_dims(img,-1)

        return img
    
class ResizeVariableWidth(object):
    """Resize image in preserving the aspect ratio (fixed height and variable width)
    """
    def __init__(self, height):
        self.height = height
        
    def __call__(self, image_array):
        
        image = image_array.copy()
        w, h = image.shape[1], image.shape[0]
        hpercent = self.height/float(h)
        wsize = int(float(w)*float(hpercent))

        image = cv2.resize(image, dsize = (wsize, self.height), interpolation = cv2.INTER_CUBIC)
        image = np.array(image, dtype = np.uint8)
#         image = np.expand_dims(image,-1)
    
        return image
        

class RandomElasticTransformer(object):
    """
    Elastic deformation of images as described in [Simard2003]_ (with modifications).
    [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
    Convolutional Neural Networks applied to Visual Document Analysis", in
    Proc. of the International Conference on Document Analysis and
    Recognition, 2003.
 
    Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
 
    def __init__(self, alpha_coeff_range, sigma_coeff_range,
                 alpha_affine_coeff_range, random_state=None):
        """
        args:
            alpha: float, multiplicative coefficient of gaussian filter
            sigma: float, parameter for scipy.ndimage.filters.gaussian_filter
            alpha_affine: float, parameter used to generate a random uniform
                          value in range [-alpha_affine, alpha_affine]
        """
        self.alpha_coeff_range = alpha_coeff_range
        self.sigma_coeff_range = sigma_coeff_range
        self.alpha_affine_coeff_range = alpha_affine_coeff_range
        self.random_state = np.random.RandomState(random_state)
 
    def __call__(self, image_array):
        """
        args:
        image_array: numpy.ndarray, representation of an image as
                         three dimensionnal vector
        """
        alpha_coeff = random.uniform(*self.alpha_coeff_range)
        sigma_coeff = random.uniform(*self.sigma_coeff_range)
        alpha_affine_coeff = random.uniform(*self.alpha_affine_coeff_range)
        image = image_array.copy()
 
        shape = image.shape
        shape_size = shape[:2]
 
        alpha = image.shape[1] * alpha_coeff
        sigma = image.shape[1] * sigma_coeff
        alpha_affine = image.shape[1] * alpha_affine_coeff
 
        # Random affine
        center_square = np.float32(shape_size) // 2
        square_size = min(shape_size) // 3
        # trick equivalent to reshape((-1, 1)) and np.vstack
        pts1 = np.float32([center_square + square_size,
                           [center_square[0] + square_size, center_square[1] - square_size],
                           center_square - square_size])
        pts2 = pts1 + self.random_state.uniform(-alpha_affine,
                                                alpha_affine,
                                                size=pts1.shape).astype(np.float32)
        M = cv2.getAffineTransform(pts1, pts2)
        image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_CONSTANT, borderValue=255).reshape(shape)
 
        dx = gaussian_filter((self.random_state.rand(*shape) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((self.random_state.rand(*shape) * 2 - 1), sigma) * alpha
 
        x, y, z = np.meshgrid(np.arange(shape[1]),
                              np.arange(shape[0]),
                              np.arange(shape[2]))
 
        indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))
 
        new_image_array = map_coordinates(image,
                                          indices,
                                          order=1,
                                          mode='constant',
                                          cval=255).reshape(shape)
        return new_image_array
    
    
class ErodePerturbation:
    """
    erodes image with OpenCV2
    """
 
    def __init__(self, range_max):
        """
        args:
            range_max: int, range of the random value generation,
                       if the generated value is above 1, the perturbation is done
        """
        self.range_max = range_max
 
    def __call__(self, image_array):
        """
        args:
            image_array: image as a numpy 3-dim array
        """
        # TODO: CHECK IF IMAGE WILL ALWAYS BE ARRAY
        new_image_array = image_array.copy()
        erode_value = random.randint(1, self.range_max)
        if erode_value > 1:
            erosion_kernel = np.ones(shape=(erode_value, erode_value))
            new_image_array = cv2.erode(new_image_array, erosion_kernel)
        return new_image_array
 
 
class RandomErodePerturbation:
    """
    erodes image with OpenCV2
    """
 
    def __init__(self, range_max_range):
        """
        args:
            range_max: int, range of the random value generation,
                       if the generated value is above 1, the perturbation is done
        """
        self.range_max_range = range_max_range
 
    def __call__(self, image_array):
        """
        args:
            image_array: image as a numpy 3-dim array
        """
        # TODO: CHECK IF IMAGE WILL ALWAYS BE ARRAY
        range_max = int(random.uniform(*self.range_max_range))
 
        try:
            erode_value = random.randint(1, range_max)
        except ValueError:
            erode_value = 1
 
        if erode_value > 1:
            new_image_array = image_array.copy()
            erosion_kernel = np.ones(shape=(erode_value, erode_value))
            new_image_array = cv2.erode(new_image_array, erosion_kernel)
            return new_image_array
        else:
            return image_array
 
 
class DilatePerturbation:
    """
    dilates image with OpenCV2
    """
 
    def __init__(self, range_max):
        """
        args:
            range_max: int, range of the random value generation,
                       if the generated value is above 1, the perturbation is done
        """
        self.range_max = range_max
 
    def __call__(self, image_array):
        """
        args:
            image_array: image as a numpy 3-dim array
        """
        # TODO: CHECK IF IMAGE WILL ALWAYS BE ARRAY
        try:
            dilate_value = random.randint(1, self.range_max)
        except ValueError:
            dilate_value = 1
 
        if dilate_value > 1:
            new_image_array = image_array.copy()
            dilation_kernel = np.ones(shape=(dilate_value, dilate_value))
            new_image_array = cv2.dilate(new_image_array, dilation_kernel)
            return new_image_array
        else:
            return image_array
 
 
class RandomDilatePerturbation:
    """
    dilates image with OpenCV2
    """
 
    def __init__(self, range_max_range):
        """
        args:
            range_max_range: int, range of the random value generation,
                             with this random value another random value is computed,
                             if this value is above 1, the perturbation is done
        """
        self.range_max_range = range_max_range
 
    def __call__(self, image_array):
        """
        args:
            image_array: image as a numpy 3-dim array
        """
        # TODO: CHECK IF IMAGE WILL ALWAYS BE ARRAY
        new_image_array = image_array.copy()
        range_max = int(random.uniform(*self.range_max_range))
        dilate_value = random.randint(1, range_max)
        if dilate_value > 1:
            dilation_kernel = np.ones(shape=(dilate_value, dilate_value))
            new_image_array = cv2.dilate(new_image_array, dilation_kernel)
        return new_image_array
 
 
class MedianBlurPerturbation:
    """
    applies a median blur perturbation on image
    """
 
    def __init__(self, size):
        """
        args:
            size: size of the median blur kernel
        """
        self.size = size
 
    def __call__(self, image_array):
        """
        args:
            image_array: image as a numpy 3-dim array
        """
        new_image = image_array.copy()
        new_image = cv2.medianBlur(new_image, self.size)
        return new_image
 
 
class RandomMedianBlurPerturbation:
    """
    applies a median blur perturbation on image
    """
 
    def __init__(self, size_range):
        """
        args:
            size: size of the median blur kernel
        """
        self.size_range = size_range
 
    def __call__(self, image_array):
        """
        args:
            image_array: image as a numpy 3-dim array
        """
        
        size = random.randint(*self.size_range)
        new_image = image_array.copy()
        new_image = cv2.medianBlur(new_image, size)
        return new_image
    
class SamplePerturbation:
    """
    applies a sampling perturbation
    """

    def __init__(self, proportion):
        """
        args:
            proportion: float, proportion of informations to drop
        """
        self.proportion = proportion

    def __call__(self, image_array):
        """
        args:
            image_array: image as a numpy 3-dim array
        """
        new_image = image_array.copy()
        # dropping white pixels
        points = (255 - new_image[:, :, 0]).nonzero()
        if points[0].shape[0] > 0:  # if image is not totally black
            sample_size = round(points[0].shape[0] * (1 - self.proportion))
            selected_indices = np.random.choice(np.arange(points[0].shape[0]),
                                                size=sample_size)
            axis_0_sample_index = points[0][selected_indices]
            axis_1_sample_index = points[1][selected_indices]
            new_image[axis_0_sample_index, axis_1_sample_index, :] = 255

        return new_image


class RandomSamplePerturbation:
    """
    applies a sampling perturbation
    """

    def __init__(self, proportion_range):
        """
        args:
            proportion: float, proportion of informations to drop
        """
        self.proportion_range = proportion_range

    def __call__(self, image_array):
        """
        args:
            image_array: image as a numpy 3-dim array
        """
        new_image = image_array.copy()
        proportion = random.uniform(*self.proportion_range)
        # dropping white pixels
        points = (255 - new_image[:, :, 0]).nonzero()
        if points[0].shape[0] > 0:  # if image is not totally black
            sample_size = round(points[0].shape[0] * (1 - proportion))
            selected_indices = np.random.choice(np.arange(points[0].shape[0]),
                                                size=sample_size)
            axis_0_sample_index = points[0][selected_indices]
            axis_1_sample_index = points[1][selected_indices]
            new_image[axis_0_sample_index, axis_1_sample_index, :] = 255

        return new_image


class StainPerturbation:
    """
    Creates local alterations with irregular sides
    """

    def __init__(self, threshold_1=0.2, threshold_2=0.7, erode_iteration=3,
                 erode_kernel_size=3):
        """
        args:
            threshold_1: float, increase this value to get more local alterations
            threshold_2: float, increate this value to get more irregular perturbations
            erode_iteration: int, number of erosion to perform
            erode_kernel: int, number of element on each axis for erosion kernel
        """
        self.threshold_1 = threshold_1
        self.threshold_2 = threshold_2
        self.erode_iteration = erode_iteration
        self.erode_kernel_size = erode_kernel_size

    def __call__(self, image_array):
        """
        args:
            image_array : np.ndarray, image as a 3-dimensionnal numpy array
        """
        new_image = image_array.copy()

        permutation_matrix_1 = np.random.random(size=new_image.shape[: 2]) <= self.threshold_1
        permutation_matrix_1 = permutation_matrix_1.astype(int) * 255
        permutation_matrix_1 = permutation_matrix_1.astype(np.uint8)

        erosion_kernel = np.ones((self.erode_kernel_size, self.erode_kernel_size))
        eroded_matrix = permutation_matrix_1
        for i in range(self.erode_iteration):
            eroded_matrix = cv2.erode(eroded_matrix, erosion_kernel)

        if len(eroded_matrix.shape) == 3:
            eroded_matrix = eroded_matrix[:, :, 0]

        permutation_matrix_2 = np.random.random(size=new_image.shape[: 2]) <= self.threshold_2
        permutation_matrix_2 = permutation_matrix_2.astype(int) * 255
        permutation_matrix_2 = permutation_matrix_2.astype(np.uint8)

        eroded_matrix[permutation_matrix_2 == 255] = 0

        new_image[eroded_matrix == 255, :] = 255 - new_image[eroded_matrix == 255, :]

        return new_image


class RandomStainPerturbation:
    """
    Creates local alterations with irregular sides
    """

    def __init__(self, threshold_1_range, threshold_2_range,
                 erode_iteration_range, erode_kernel_size_range):
        """
        args:
            threshold_1: float, increase this value to get more local alterations
            threshold_2: float, increate this value to get more irregular perturbations
            erode_iteration: int, number of erosion to perform
            erode_kernel: int, number of element on each axis for erosion kernel
        """
        self.threshold_1_range = threshold_1_range
        self.threshold_2_range = threshold_2_range
        self.erode_iteration_range = erode_iteration_range
        self.erode_kernel_size_range = erode_kernel_size_range

    def __call__(self, image_array):
        """
        args:
            image_array : np.ndarray, image as a 3-dimensionnal numpy array
        """
        threshold_1 = random.uniform(*self.threshold_1_range)
        threshold_2 = random.uniform(*self.threshold_2_range)
        erode_iteration = int(random.uniform(*self.erode_iteration_range))
        erode_kernel_size = int(random.uniform(*self.erode_kernel_size_range))
        new_image = image_array.copy()

        permutation_matrix_1 = np.random.random(size=new_image.shape[: 2]) <= threshold_1
        permutation_matrix_1 = permutation_matrix_1.astype(int) * 255
        permutation_matrix_1 = permutation_matrix_1.astype(np.uint8)

        erosion_kernel = np.ones((erode_kernel_size, erode_kernel_size))
        eroded_matrix = permutation_matrix_1
        for i in range(erode_iteration):
            eroded_matrix = cv2.erode(eroded_matrix, erosion_kernel)

        if len(eroded_matrix.shape) == 3:
            eroded_matrix = eroded_matrix[:, :, 0]

        permutation_matrix_2 = np.random.random(size=new_image.shape[: 2]) <= threshold_2
        permutation_matrix_2 = permutation_matrix_2.astype(int) * 255
        permutation_matrix_2 = permutation_matrix_2.astype(np.uint8)

        eroded_matrix[permutation_matrix_2 == 255] = 0

        new_image[eroded_matrix == 255, :] = 255 - new_image[eroded_matrix == 255, :]

        return new_image


class ResizePerturbation:
    """
    downsizes image and put it back in the original size in way to
    loose informations
    """

    def __init__(self, transformation_factor=0.3, binarizer_threshold=200):

        self.factor = random.random() * (1 - transformation_factor) + transformation_factor
        self.pil_to_numpy = PILToNumpy()
        self.numpy_to_pil = NumpyToPIL()
        self.binarizer = ImageBinarizer(binarizer_threshold)

    def __call__(self, image_array):
        """
        args:
            image_array : np.ndarray, image as a 3-dimensionnal numpy array
        """
        new_image_array = image_array.copy()

        original_size = image_array.shape[:2][::-1]

        new_size = (int(image_array.shape[1] * self.factor),
                    int(image_array.shape[0] * self.factor))

        image_PIL = self.numpy_to_pil(new_image_array)
        image_PIL = image_PIL.resize(new_size, resample=Image.BILINEAR)

        new_image_array = self.pil_to_numpy(image_PIL)
        new_image_array = self.binarizer(new_image_array)

        image_PIL = self.numpy_to_pil(new_image_array)
        image_PIL = image_PIL.resize(original_size, resample=Image.BILINEAR)

        new_image_array = self.pil_to_numpy(image_PIL)
        new_image_array = self.binarizer(new_image_array)

        return new_image_array


class RandomResizePerturbation:
    """
    downsizes image and put it back in the original size in way to
    loose informations
    """

    def __init__(self, transformation_factor_range, binarizer_threshold_range):

        self.transformation_factor_range = transformation_factor_range
        self.pil_to_numpy = PILToNumpy()
        self.numpy_to_pil = NumpyToPIL()
        self.binarizer = RandomImageBinarizer(binarizer_threshold_range)

    def __call__(self, image_array):
        """
        args:
            image_array : np.ndarray, image as a 3-dimensionnal numpy array
        """
        transformation_factor = random.uniform(*self.transformation_factor_range)
        factor = random.random() * (1 - transformation_factor) + transformation_factor
        new_image_array = image_array.copy()

        original_size = image_array.shape[:2][::-1]

        new_size = (int(image_array.shape[1] * factor),
                    int(image_array.shape[0] * factor))

        image_PIL = self.numpy_to_pil(new_image_array)
        image_PIL = image_PIL.resize(new_size, resample=Image.BILINEAR)

        new_image_array = self.pil_to_numpy(image_PIL)
        new_image_array = self.binarizer(new_image_array)

        image_PIL = self.numpy_to_pil(new_image_array)
        image_PIL = image_PIL.resize(original_size, resample=Image.BILINEAR)

        new_image_array = self.pil_to_numpy(image_PIL)
        new_image_array = self.binarizer(new_image_array)

        return new_image_array


class NumpyToPIL:
    """
    Converter perturbation like to allow integration in torch's Compose
    """

    def __init__(self):
        pass

    def __call__(self, image_array):
        """
        image array : numpy.ndarray, image as array
        """
        image_pil = Image.fromarray(image_array)
        return image_pil


class PILToNumpy:
    """
    Converter perturbation like to allow integration in torch's Compose
    """

    def __init__(self):
        pass

    def __call__(self, image_pil):
        """
        image array : numpy.ndarray, image as array
        """
        image_array = np.array(image_pil)
        return image_array
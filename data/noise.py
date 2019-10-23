from lib import *

def random_num_generator(random_config, random_state = np.random):
    if random_config[0] == 'uniform':
        ret = random_state.uniform(random_config[1], random_config[2])
    elif random_config[0] == 'lognormal':
        ret = random_state.lognormal(random_config[1], random_config[2])
    else:
        raise Exception('unsupported format {format}. Must be "uniform" for "lognormal"'.format(format = random_config[0]))        
    return ret

class GaussianNoise(object):
    
    def __init__(self, mean, sigma, random_state = np.random):
        self.sigma = sigma
        self.mean = mean
        self.random_state = random_state
        
    def __call__(self, image):
        
        if isinstance(self.sigma, collections.Sequence):
            sigma = random_num_generator(self.sigma, random_state = self.random_state)
        else:
            sigma = self.sigma
        
        if isinstance(self.mean, collections.Sequence):
            mean = random_num_generator(self.mean, random_state = self.random_state)
        else:
            mean = self.mean
        
        row, col, ch = image.shape
        gauss = self.random_state.normal(mean, sigma, (row, col, ch))
#         gauss = gauss.reshape(row, col, ch)
        image_new = image + gauss
        return image
    
class SpeckleNoise(object):
    
    def __init__(self, mean, sigma, random_state = np.random):
        self.mean = mean
        self.sigma = sigma
        self.random_state = random_state
    
    def __call__(self, image):
        
        if isinstance(self.sigma, collections.Sequence):
            sigma = random_num_generator(self.sigma, random_state = self.random_state)
        else:
            sigma = self.sigma
        
        if isinstance(self.mean, collections.Sequence):
            mean = random_num_generator(self.mean, random_state = self.random_state)
        else:
            mean = self.mean
        
#         row, col, ch = image.shape
        gauss = self.random_state.normal(mean, sigma, image.shape)
#         gauss = gauss.reshape(row, col, ch)
        image_new = image + image * gauss
        return image_new

def poisson_downsampling(image, peak, random_state = np.random):
    if not isinstance(image, np.ndarray):
        imgArr = np.array(image, dtype = 'float32')
    else:
        imgArr = image.astype('float32')
    Q = imgArr.max(axis=(0, 1))/peak
    if Q[0] == 0:
        return imgArr
    ima_lambda = imgArr / Q
    noisy_img = random_state.poisson(lam = ima_lambda)
    return noisy_img.astype('float32')
    
class GaussianPoissonNoise(object):
    
    def __init__(self, sigma, peak, random_state = np.random):
        self.sigma = sigma
        self.peak = peak
        self.random_state = random_state
    
    def __call__(self, image):
        
        if isinstance(self.peak, collections.Sequence):
            peak = random_num_generator(self.peak, random_state = self.random_state)
        else:
            peak = self.peak
        
        if isinstance(self.sigma, collections.Sequence):
            sigma = random_num_generator(self.sigma, random_state = self.random_state)
        else:
            sigma = self.sigma
        
        bg = gaussian_filter(image, sigma = (sigma, sigma, 0))
        bg = poisson_downsampling(bg, peak = peak, random_state = self.random_state)
        image_new = image + bg
        
        return image_new
    
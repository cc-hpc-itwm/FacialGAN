#from torchvision.datasets.vision import VisionDataset
from core.VisionDataset import VisionDataset

from PIL import Image

import os
import os.path

import imageio
import numpy as np
import random

from torchvision import transforms
import torch



def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def is_image_file(filename):
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def make_dataset(directory, class_to_idx, extensions=None, is_valid_file=None):
    instances = []
    directory = os.path.expanduser(directory)
    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x):
            return has_file_allowed_extension(x, extensions)
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = path, class_index
                    instances.append(item)
    return instances


class DatasetFolder(VisionDataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root, root_masks, loader, attributes, training, extensions=None, transform=None,
                 target_transform=None, is_valid_file=None):
        super(DatasetFolder, self).__init__(root, root_masks, attributes, training, transform=transform,
                                            target_transform=target_transform)
        classes, class_to_idx = self._find_classes(self.root)
        samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"
                                "Supported extensions are: " + ",".join(extensions)))

        self.root_masks = root_masks
        self.loader = loader
        self.training = training
        self.extensions = extensions
        self.attributes = attributes

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """        
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx
    
    def random_attribute(self, fname, size, attributes):
        
        width, height = size
        sample_mask = np.zeros([width//2,height//2,3])
        sel_attributes = []
        
        for i in range (len(attributes)):
  
            if attributes[i] == "eyes":
                sel_attributes.append("l_eye")
                sel_attributes.append("r_eye")
                sel_attributes.append("l_brow")
                sel_attributes.append("r_brow")
            elif attributes[i] == "lip":
                sel_attributes.append("l_lip")
                sel_attributes.append("u_lip")
                sel_attributes.append("mouth")
            else:
                sel_attributes.append(attributes[i])
        
        for i in range (len(sel_attributes)):

            mask_file = fname+"_"+sel_attributes[i]+".png"                
            mask_path = os.path.join(self.root_masks, mask_file)

            if os.path.isfile(mask_path):
                img = imageio.imread(mask_path)
                sample_mask = sample_mask + img
                    
        return sample_mask
                                          
    def _find_masks(self, path, size):
        """
        Args:
            path (strg): Index
            size(tuple): Image size

        Returns:
            mask
        """
        fname = path[-9:-4]
        try:
            attributes = self.attributes.split(", ")
            mask_eyes = self.random_attribute(fname, size, [attributes[0]])
            mask_lip = self.random_attribute(fname, size, [attributes[1]]) 
            mask_nose = self.random_attribute(fname, size, [attributes[2]])
            mask_skin = self.random_attribute(fname, size, ["skin"])
        except: 
            pass
        
        mask_eyes = Image.fromarray(np.uint8(mask_eyes))
        mask_lip = Image.fromarray(np.uint8(mask_lip))
        mask_nose = Image.fromarray(np.uint8(mask_nose))
        mask_skin = Image.fromarray(np.uint8(mask_skin))

        return mask_eyes, mask_lip, mask_nose, mask_skin

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
                
        mask_eyes, mask_lip, mask_nose, mask_skin = self._find_masks(path, sample.size)
     
        if self.training == True:
            if random.random() > 0.5:
                sample = transforms.functional.hflip(sample)
                mask_eyes = transforms.functional.hflip(mask_eyes)
                mask_lip = transforms.functional.hflip(mask_lip)
                mask_nose = transforms.functional.hflip(mask_nose)
                mask_skin = transforms.functional.hflip(mask_skin)

        if self.transform is not None:
            sample = self.transform(sample)
            mask_eyes = self.transform(mask_eyes)
            mask_lip = self.transform(mask_lip)
            mask_nose = self.transform(mask_nose)
            mask_skin = self.transform(mask_skin)
            
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        sample_mask = torch.ones([4, mask_eyes.shape[1], mask_eyes.shape[2]])
        sample_mask[0] = mask_eyes[0]
        sample_mask[1] = mask_lip[0]
        sample_mask[2] = mask_nose[0]
        sample_mask[3] = mask_skin[0]

        return sample, target, sample_mask

    def __len__(self):
        return len(self.samples)


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class CustomImageFolder(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, root_masks, attributes, training, transform=None, target_transform=None,
                 loader=default_loader, is_valid_file=None):
        super(CustomImageFolder, self).__init__(root, root_masks, loader, attributes, training,
                                                IMG_EXTENSIONS if is_valid_file is None else None,
                                                transform=transform, target_transform=target_transform,
                                                is_valid_file=is_valid_file)
        self.imgs = self.samples

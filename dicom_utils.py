'''
Filename: dicom_utils.py
Author(s): Jonathan Burkow, burkowjo@msu.edu, Michigan State University
           Greg Holste, giholste@gmail.com, UT Austin
Last Updated: 03/31/2022
Description: A collection of utility functions needed to process through DICOM files, including
    thresholding, cropping, histogram qualization, and saving to PNGs.
'''

import math
import os
from typing import Dict, List, Tuple

import cv2
import numpngw
import numpy as np
import pydicom
import torch
from skimage import color, exposure, filters

from ChestSeg_PyTorch.preprocess import to_one_hot
from ChestSeg_PyTorch.utils import postprocess

# Globally define U-Net output classes
CLASSES = ['background', 'spine', 'mediastinum', 'left_lung', 'right_lung', 'left_subdiaphragm', 'right_subdiaphragm']


def invert_image(image: np.ndarray) -> np.ndarray:
    """
    If image has reversed intensity values (i.e., values close to 0 appear white and higher appear
    black) invert array values so values close to 0 appear black and higher appear white.

    Notes
    -----
    Subtracting by the min value retains the same value range as the original image.
    """
    return np.invert(image) - np.invert(image).min()


def load_dicom_image(dicom_file: pydicom.dataset.Dataset) -> np.ndarray:
    """
    Load in image from the DICOM file, apply necessary scaling, and return array.

    Parameters
    ----------
    dicom_file : the DICOM file read in by PyDicom
    """
    # Pull out the image array from the dicom file.
    # Depending on DICOM Photometric Interpretation value, invert the image.
    # RGB : convert to 1-channel grayscale
    # MONOCHROME1 : values close to 0 appear white => invert
    # MONOCHROME2 : values close to 0 appear black => keep original
    if dicom_file.PhotometricInterpretation == 'RGB':
        image = color.rgb2gray(dicom_file.pixel_array.copy())
    elif dicom_file.PhotometricInterpretation == 'MONOCHROME1':
        image = invert_image(dicom_file.pixel_array.copy().astype('uint16'))
    else:
        image = dicom_file.pixel_array.copy()

    # Retrieve pixel intensity data from the dicom file
    # Rescale array values to ensure pixel intensities reflect original data
    slope = dicom_file.RescaleSlope if hasattr(dicom_file, 'RescaleSlope') else 1.0
    intercept = dicom_file.RescaleIntercept if hasattr(dicom_file, 'RescaleIntercept') else 0.0
    image = slope * image + intercept

    return image.astype(float)


def threshold_image(
        image: np.ndarray,
        method: str = 'li'
    ) -> np.ndarray:
    """
    Use scikit-image filters to threshold image array (set all values below a certain value to zero)

    Parameters
    ----------
    image  : array of the DICOM image
    method : the method of thresholding to use

    Returns
    -------
    image : array of the image after the thresholding is applied
    """
    # Set the threshold value based on specified method
    if method == 'otsu':
        threshold = filters.threshold_otsu(image)
    elif method == 'multi_otsu':
        thresholds = filters.threshold_multiotsu(image)
        threshold = thresholds[0]
    elif method == 'li':
        threshold = filters.threshold_li(image)

    # Threshold the image
    image[image < threshold] = 0

    return image


def rough_crop(
        image: np.ndarray,
        blank_dist: int = 20
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    The first stage of cropping for processing DICOM images. Takes in numpy array representation of
    DICOM image, crops array based on row/column sums with nonzero elements. Outputs the x and y
    offsets for adjusting annotations post-crop.

    Parameters
    ----------
    image      : array of the DICOM image
    blank_dist : distance between nonzero indices to determine crop boundaries

    Returns
    -------
    image   : array of the image after rough cropping is performed
    indices : tuple containing the boundary indices for the rough crop; (top, bottom, left, right)
    """
    # Extract image height and width
    h, w = image.shape[0], image.shape[1]

    # Compute row and column sums and find nonzero indices
    col_nonzero = image.sum(axis=0).nonzero()[0]
    row_nonzero = image.sum(axis=1).nonzero()[0]

    # Create dummy variables for each bound
    left_bound, right_bound, top_bound, bottom_bound = -1, -1, -1, -1

    # Compute left bound based on black space between indices
    for i in range(len(col_nonzero) - 1):
        if col_nonzero[i] > w // 2:
            break
        if col_nonzero[i+1] - col_nonzero[i] > blank_dist:
            left_bound = col_nonzero[i+1]
    # Set left bound to min nonzero index if no black space exists
    left_bound = col_nonzero.min() if left_bound < 0 else left_bound

    # Compute right bound based on black space between indices
    for i in range(len(col_nonzero)-1, 0, -1):
        if col_nonzero[i] < w // 2:
            break
        if col_nonzero[i] - col_nonzero[i-1] > blank_dist:
            right_bound = col_nonzero[i-1]
    # Set right bound to min nonzero index if no black space exists
    right_bound = col_nonzero.max() if right_bound < 0 else right_bound

    # Compute top bound based on black space between indices
    for i in range(len(row_nonzero) - 1):
        if row_nonzero[i] > h // 2:
            break
        if row_nonzero[i+1] - row_nonzero[i] > blank_dist:
            top_bound = row_nonzero[i+1]
    # Set top bound to min nonzero index if no black space exists
    top_bound = row_nonzero.min() if top_bound < 0 else top_bound

    # Compute bottom bound based on black space between indices
    for i in range(len(row_nonzero)-1, 0, -1):
        if row_nonzero[i] < h // 2:
            break
        if row_nonzero[i] - row_nonzero[i-1] > blank_dist:
            bottom_bound = row_nonzero[i-1]
    # Set bottom bound to min nonzero index if no black space exists
    bottom_bound = row_nonzero.max() if bottom_bound < 0 else bottom_bound

    # Store the rough crop indices
    top = int(top_bound)
    bottom = int(bottom_bound) + 1
    left = int(left_bound)
    right = int(right_bound) + 1

    indices = (top, bottom, left, right)

    # Crop the original image based on rough crop indices
    image = image[top:bottom, left:right]

    return image, indices


def quadrant_indices(
        quadrant_array: np.ndarray,
        quad_height: int,
        quad_width: int,
        quadrant: str,
        sum_threshold: float = 0.10
    ) -> Tuple[int, int]:
    """
    For a given quadrant of the image array, return corresponding indices for the boundaries
    obtained for the second stage of cropping.

    Parameters
    ----------
    quadrant_array : the quadrant of the image
    quad_height    : height (# of rows) of the quadrant
    quad_width     : width (# of columns) of the quadrant
    quadrant       : text specifying which quadrant to get indices for
    sum_threshold  : % of pixels that must contain information to be kept within crop boundaries

    Returns
    -------
    Combination of left/top, left/bottom, right/top, or right/bottom indices to adjust crop on image
    """
    # Convert array to binary (pixels with data = 1, else 0)
    binary = quadrant_array > 0

    # Calculate row and column sums of binary array, find indices with > threshold #pixels with data
    col_binary = binary.sum(axis=0) > quad_height * sum_threshold
    row_binary = binary.sum(axis=1) > quad_width * sum_threshold

    if quadrant == 'topleft':
        try:
            left_bound = col_binary.nonzero()[0].min()
        except:
            left_bound = 0
        try:
            top_bound = row_binary.nonzero()[0].min()
        except:
            top_bound = 0
        return int(left_bound), int(top_bound)

    if quadrant == 'topright':
        try:
            right_bound = col_binary.nonzero()[0].max()
        except:
            right_bound = quad_width
        try:
            top_bound = row_binary.nonzero()[0].min()
        except:
            top_bound = 0
        return int(right_bound), int(top_bound)

    if quadrant == 'bottomleft':
        try:
            left_bound = col_binary.nonzero()[0].min()
        except:
            left_bound = 0
        try:
            bottom_bound = row_binary.nonzero()[0].max()
        except:
            bottom_bound = quad_height
        return int(left_bound), int(bottom_bound)

    if quadrant == 'bottomright':
        try:
            right_bound = col_binary.nonzero()[0].max()
        except:
            right_bound = quad_width
        try:
            bottom_bound = row_binary.nonzero()[0].max()
        except:
            bottom_bound = quad_height
        return int(right_bound), int(bottom_bound)

    if quadrant == 'centerleft':
        try:
            left_bound = np.where(col_binary == 0)[0][-1]
            return int(left_bound)
        except:
            return 0

    if quadrant == 'centerright':
        try:
            right_bound = np.where(col_binary == 0)[0][0]
            return int(right_bound)
        except:
            return quad_width


def get_quad_crop_offsets(image: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Second stage option of cropping DICOM image by splitting the array into quadrants and looking at
    empty space within each. After indices are found from each quadrant, the tightest bounds are
    chosen and the image is cropped with them, plus an additional amount given by pixel_crop.

    Parameters
    ----------
    image   : array of the DICOM image after initial crop

    Returns
    -------
    pixel offsets to constrict after rough crop; (top, bottom, left, right)
    """
    # Get the height and width of the quadrants
    quad_h, quad_w = image.shape[0]//2, image.shape[1]//2

    ### Top Left Quadrant ###
    top_left = image[:quad_h, :quad_w]
    # Get left and top indices from top left quadrant
    tl_left_bound, tl_top_bound = quadrant_indices(top_left, quad_h, quad_w, 'topleft')

    ### Top Right Quadrant ###
    top_right = image[:quad_h, quad_w:]
    # Get right and top indices from top right quadrant
    tr_right_bound, tr_top_bound = quadrant_indices(top_right, quad_h, quad_w, 'topright')

    ### Bottom Left Quadrant ###
    bottom_left = image[:quad_h, :quad_w]
    # Get left and bottom indices from bottom left quadrant
    bl_left_bound, bl_bottom_bound = quadrant_indices(bottom_left, quad_h, quad_w, 'bottomleft')

    ### Bottom Right Quadrant ###
    bottom_right = image[quad_h:, quad_w:]
    # Get right and bottom indices from top left quadrant
    br_right_bound, br_bottom_bound = quadrant_indices(bottom_right, quad_h, quad_w, 'bottomright')

    # Calculate overall left, top, right, and bottom bounds
    top_bound = int(max(tl_top_bound, tr_top_bound))
    bottom_bound = int(image.shape[0] - (min(bl_bottom_bound, br_bottom_bound) + quad_h))
    left_bound = int(max(tl_left_bound, bl_left_bound))
    right_bound = int(image.shape[1] - (min(tr_right_bound, br_right_bound) + quad_w))

    return top_bound, bottom_bound, left_bound, right_bound


def get_center_crop_offsets(image: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Second stage option of cropping the DICOM image by looking at the middle 50% region of the
    image, splitting it into left and right sections, and finding empty space within each.
    After indices are found from each quadrant, the tightest bounds are chosen and the image is
    cropped with them plus an additional amount given by pixel_crop.

    Parameters
    ----------
    image : array of the DICOM image after initial crop

    Returns
    -------
    pixel offsets to constrict after rough crop; (top, bottom, left, right)
    """
    # Get the height and width of the center region
    center_h, center_w = image.shape[0]//2, image.shape[1]//2

    # Get center top and bottom indices
    center_top, center_bottom = image.shape[0]//4, 3*(image.shape[0]//4)

    # Get left bound
    center_left = image[center_top:center_bottom, :center_w]
    left_bound = quadrant_indices(center_left, center_h, center_w, 'centerleft')

    # Get right bound
    center_right = image[center_top:center_bottom, center_w:]
    right_bound = quadrant_indices(center_right, center_h, center_w, 'centerright')

    # Get top bound
    top_left = image[:center_h, :center_w]
    _, tl_top_bound = quadrant_indices(top_left, center_h, center_w, 'topleft')
    top_right = image[:center_h, center_w:]
    _, tr_top_bound = quadrant_indices(top_right, center_h, center_w, 'topright')

    # Get bottom bound
    bottom_left = image[:center_h, :center_w]
    _, bl_bottom_bound = quadrant_indices(bottom_left, center_h, center_w, 'bottomleft')
    bottom_right = image[center_h:, center_w:]
    _, br_bottom_bound = quadrant_indices(bottom_right, center_h, center_w, 'bottomright')

    # Calculate overall left, top, right, and bottom bounds
    top_bound = int(max(tl_top_bound, tr_top_bound))
    bottom_bound = int(image.shape[0] - (min(bl_bottom_bound, br_bottom_bound) + center_h))
    left_bound = int(left_bound)
    right_bound = int(image.shape[1] - (right_bound + center_w))

    return top_bound, bottom_bound, left_bound, right_bound


def get_unet_offsets(cat_mask: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Find offsets for predicted segmentation mask at rough crop dimensions.

    Parameters
    ----------
    cat_mask : categorical segmentation mask of shape (w, h)

    Returns
    -------
    offsets for tighter cropping of rough-cropped image and mask; (top, bottom, left and right)
    """
    # Find indices where non-background pixels appear
    fg_indices = np.nonzero(cat_mask)

    # Find top, bottom, left, and right offsets from rough-cropped boundaries
    top_offset = fg_indices[0].min()
    bottom_offset = cat_mask.shape[0] - fg_indices[0].max() + 1
    left_offset = fg_indices[1].min()
    right_offset = cat_mask.shape[1] - fg_indices[1].max() + 1

    return top_offset, bottom_offset, left_offset, right_offset


def unet_crop(
        image: np.ndarray,
        model: torch.nn.Module,
        device: torch.device,
        verbose: bool = False
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Second stage option of cropping the DICOM image by using a trained U-Net network to find an
    instance segmentation of the thoracic cavity and ribs and finding the smallest and largest
    indices of those masks.

    Parameters
    ----------
    image   : array of the DICOM image after initial crop
    model   : PyTorch U-Net model; use segmentation instead of region crop for second crop stage
    device  : PyTorch device (cuda or cpu)
    verbose : If True, print out the index offsets using center cropping

    Returns
    -------
    cat_y_pred : processed, categorical segmentation mask of shape (h, w)
    offsets    : pixel offsets to constrict after rough crop; (top, bottom, left, right)
    """
    # Make copy of image to prep for U-Net
    image_copy = image.copy()

    # Resize and normalize to [0, 1] for U-Net
    image_copy = cv2.resize(image_copy, (256, 256))
    image_copy = (image_copy - np.min(image_copy)) / (np.max(image_copy) - np.min(image_copy)) 

    # Convert image to tensor
    image_copy = torch.Tensor(image_copy).unsqueeze(0).unsqueeze(0).to(device)

    # Predict segementation mask with U-Net
    y_hat = torch.stack([y_h for y_h in model.forward(image_copy)], dim=0).mean(dim=0)

    # Apply post-processing and threshold to binary (one-hot) predicted segmentation
    proc_y_pred = postprocess(y_hat, CLASSES).squeeze()

    # Convert to categorical and resize to original rough crop dimensions
    # cat_y_pred = cv2.resize(proc_y_pred.argmax(axis=1), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    cat_y_pred = cv2.resize(proc_y_pred.argmax(axis=0), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Calculate offsets for final image cropping based on U-Net segmentation
    offsets = get_unet_offsets(cat_y_pred)

    # Print second stage crop offsets
    if verbose:
        print('U-Net segmentation crop offsets:', offsets)

    return cat_y_pred, offsets


def crop_dicom(
        dicom_file: pydicom.Dataset,
        verbose: bool = False,
        crop_region: str = 'center',
        model: torch.nn.Module = None,
        device: torch.device = None
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Full function to crop a DICOM image. The pixel array is cropped through two stages:
        1) a rough crop to center on the X-Ray
        2) eliminates glowing/white borders and as much empty space as possible.
    The offsets are preserved for adjusting annotations later.

    Parameters
    ----------
    dicom_file  : DICOM structure of the current image
    verbose     : If True, print out each stage of image processing steps
    crop_region : determines second stage of cropping if model is None; (choice: quadrant, center)
    model       : PyTorch U-Net model; use segmentation instead of region crop for second crop stage
    device      : PyTorch device (cuda or cpu)

    Returns
    -------
    y_pred  : fully processed and cropped segmentation mask
    indices : the pixel offsets needed to adjust annotations; (top, bottom, left, right)
    """
    # Make a copy of the image to modify
    image = load_dicom_image(dicom_file)
    image_copy = image.copy()
    orig_image_shape = image.shape

    # Threshold the image copy for improved cropping
    if 'fracture_unknown_0436' not in dicom_file.PatientName:
        image_copy = threshold_image(image_copy, method='li')

        # Get the indices for the rough crop stage
        image_copy, init_crop = rough_crop(image_copy)
    else: # 0436 is an outlier surrounded by white, messes up thresholding and thus the cropping
        init_crop = (1150, 2560, 0, 1250)
        image_copy = image[init_crop[0]:init_crop[1], init_crop[2]:init_crop[3]]

    # Get the index offsets from the region crop stage or U-Net segmentation
    if model is not None:
        cat_y_pred, region_offsets = unet_crop(image_copy, model, device, verbose=verbose)
    elif crop_region == 'center':
        region_offsets = get_center_crop_offsets(image_copy)
    elif crop_region == 'quad':
        region_offsets = get_quad_crop_offsets(image_copy)

    # Calculate rough crop + unet/region crop indices
    indices = (max(0, init_crop[0] + region_offsets[0]),
               min(image.shape[0], init_crop[1] - region_offsets[1]),
               max(0, init_crop[2] + region_offsets[2]),
               min(image.shape[1], init_crop[3] - region_offsets[3]))

    if model is not None:
        # Pad pred array with zeros based on rough crop indices to match original image shape
        cat_y_pred = np.pad(cat_y_pred, [(init_crop[0], orig_image_shape[0] - init_crop[1]),
                                         (init_crop[2], orig_image_shape[1] - init_crop[3])])

        # Convert to one-hot (H, W, n_classes)
        y_pred = to_one_hot(cat_y_pred)

        return y_pred, indices
    else:
        return indices


def save_to_npy(y_pred: np.ndarray, save_loc: str) -> None:
    """
    Save NumPy array to a .npy file.

    Parameters
    ----------
    y_pred   : fully processed, cropped segmentation mask
    save_loc : filepath of destination of image_array
    """
    np.save(save_loc, y_pred)


def create_rgb(image: np.ndarray) -> np.ndarray:
    """
    Returns a three-channel (RGB) stacked array of the input image array.

    Parameters
    ----------
    image : single-channel array of the image
    """
    return np.stack([image, image, image], axis=-1)


def save_to_png(
        image_array: np.ndarray,
        save_loc: str,
        overwrite: bool = False
    ) -> None:
    """
    Save the image array to a RGB PNG file.

    Parameters
    ----------
    image_array : the fully processed DICOM image
    save_loc    : directory location and filename to save the PNG to
    overwrite   : True/False on whether to overwrite an existing image
    """
    if not overwrite and not os.path.exists(save_loc) or overwrite:
        numpngw.write_png(save_loc, image_array)


def scale_image_to_depth(
        image: np.ndarray,
        bit_depth: int
    ) -> np.ndarray:
    """
    Takes the input image array and scales the values to the specified bit depth.

    Parameters
    ----------
    image     : array of the image
    bit_depth : bit depth the image will be saved as
    """
    # Normalize all values between [0,1] if they aren't already
    image = image if image.max() == 1.0 else (image / image.max())

    # Scale values to specified bit depth values
    image *= 2**bit_depth - 1

    # Return image as specific datatype based on bit depth
    if bit_depth == 8:
        return image.astype('uint8')

    if bit_depth == 16:
        return image.astype('uint16')


def hist_equalization(
        image: np.ndarray,
        method: str = 'hand',
        bit_depth: int = 16
    ) -> np.ndarray:
    """
    Perform histogram equalization on the input image, dependent on the method chosen. OpenCV
    requires the image to be 8-bit. Sk-image outputs an array with values between [0,1].

    Parameters
    ----------
    image     : array of the image
    method    : method to perform histogram equalization; (choices: 'opencv', 'skimage', 'hand')
    bit_depth : max bit depth of the array to equalize to; only used if method=='hand'

    Returns
    -------
    hist_eq_img : histogram equalized image
    """
    if method == 'opencv':
        # Convert array to 8-bit
        if image.dtype != 'uint8':
            image = scale_image_to_depth(image, 8)
        # Equalize the image
        hist_eq_img = cv2.equalizeHist(image)

    elif method == 'skimage':
        hist_eq_img = exposure.equalize_hist(image)

    elif method == 'hand':
        # Set the number of intensity values for the image image
        L = 2**bit_depth  # 65536 for 16, 256 for 8

        if L < image.max():
            image /= image.max()
            image *= L

        # Calculate the PMF and CDF of the original cropped image
        hist, bins = np.histogram(image.flatten(), bins=L, range=[0, (L-1)])
        # pixel_pdf = hist / image.size
        pixel_cdf = hist.cumsum()

        # Get the equalized values for the pixel intensities
        eq = pixel_cdf / image.size * (L - 1)

        # Apply the equalization to the original intensities
        hist_eq_img = np.interp(image.flatten(), bins[:-1], eq)
        # Reshape the array to the original shape
        hist_eq_img = hist_eq_img.reshape(image.shape)

    return hist_eq_img


def extract_bboxes(annotation_data: Dict[str, str]) -> Tuple[List[int], List[int], List[int], List[int]]:
    """
    Pull out the top left and bottom right (x,y) locations for boundary boxes.

    Parameters
    ----------
    annotation_data : information for the DICOM image annotations

    Returns
    -------
    tl_xs : list of top left boundary box x-values
    tl_ys : list of top left boundary box y-values
    br_xs : list of bottom right boundary box x-values
    br_ys : list of bottom right boundary box y-values
    """
    tl_xs, tl_ys, br_xs, br_ys = [], [], [], []
    for pt in zip(annotation_data['rectangles']['start_points'],
                  annotation_data['rectangles']['end_points']):
        x1 = math.floor(pt[0][0])
        y1 = math.floor(pt[0][1])
        x2 = math.ceil(pt[1][0])
        y2 = math.ceil(pt[1][1])

        # If box was made from right to left, flip x values
        if x2 < x1:
            x1, x2 = x2, x1
        # If box was made from bottom to top, flip y values
        if y2 < y1:
            y1, y2 = y2, y1

        tl_xs.append(x1)
        tl_ys.append(y1)
        br_xs.append(x2)
        br_ys.append(y2)

    return tl_xs, tl_ys, br_xs, br_ys

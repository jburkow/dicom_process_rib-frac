'''
Filename: dicom_utils.py
Authors: Jonathan Burkow, burkowjo@msu.edu
         Greg Holste, holstegr@msu.edu
         Michigan State University
Last Updated: 09/06/2020
Description: A collection of utility functions needed to process through
    DICOM files, including thresholding, cropping, histogram
    equalization, and saving to PNGs.
'''

import os
import csv
import math
import numpy as np
import cv2
from skimage import exposure, color, filters
import numpngw
from unet_utils import to_binary, to_one_hot, postprocess, get_unet_offsets
from general_utils import plot_image

def load_dicom_image(dicom_file, verbose=False):
    """
    Load in the image array from the dicom file.

    Parameters
    ----------
    dicom_file : PyDICOM dataset
        the DICOM file read in by PyDicom
    verbose : bool
        If True, print out the original image

    Returns
    -------
    image : ndarray
        the image array
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
    try:
        slope = float(dicom_file.RescaleSlope)
        intercept = float(dicom_file.RescaleIntercept)
    except:
        slope = 1.0
        intercept = 0.0
    image = slope * image + intercept

    # Plot the image from the dicom file
    if verbose: plot_image(image, title='Original Image')

    return image.astype(float)

def invert_image(image):
    """
    If the image has reversed intensity values, i.e., values close to 0
    appear white and higher appear black, invert the image array values
    so that values close to 0 appear black and higher appear white.

    Notes
    -----
    Subtracting by the min value retains the same value range as the
    original image.
    """
    return np.invert(image) - np.invert(image).min()



def threshold_image(image, method='li', verbose=False):
    """
    Use scikit-image filtering methods to threshold the image array to
    set all values below a certain value to zero. Used for cropping.

    Parameters
    ----------
    image : ndarray
        array of the DICOM image
    method : str
        the method of thresholding to use
    verbose : bool
        If True, print out the thresholded image

    Returns
    -------
    image : ndarray
        array of the image after the thresholding is applied
    """
    # Find the threshold based on method
    if method == 'otsu':
        threshold = filters.threshold_otsu(image)
    elif method == 'multi_otsu':
        thresholds = filters.threshold_multiotsu(image)
        threshold = thresholds[0]
    elif method == 'li':
        threshold = filters.threshold_li(image)

    # Threshold the image
    image[image < threshold] = 0

    # Plot the current stage of the image
    if verbose: plot_image(image, title='Thresholded Image')

    return image

def rough_crop(image, blank_dist=20, verbose=False):
    """
    The first stage of cropping for processing DICOM images. Taking in
    a numpy array representation of the DICOM image, crop the array
    based on row/column sums with nonzero elements. Outputs the x and y
    offsets for adjusting annotations post-crop.

    Parameters
    ----------
    image : ndarray
        array of the DICOM image
    blank_dist : int
        distance between nonzero indices to determine crop boundaries
    verbose : bool
        If True, print out the cropped image

    Returns
    -------
    image : ndarray
        array of the image after rough cropping is performed
    indices : int, (top, bottom, left, right)
        tuple containing the boundary indices for the rough crop
    """
    # Extract image height and width
    h, w = image.shape[0], image.shape[1]

    # Compute row and column sums and find nonzero indices
    col_nonzero = image.sum(axis=0).nonzero()[0]
    row_nonzero = image.sum(axis=1).nonzero()[0]

    # Create dummy variables for each bound
    left_bound, right_bound, top_bound, bottom_bound = -1, -1, -1, -1

    # Compute left bound based on black space between indices
    for i in range(0, len(col_nonzero) - 1):
        if col_nonzero[i] > w // 2:
            break
        if col_nonzero[i+1] - col_nonzero[i] > blank_dist:
            left_bound = col_nonzero[i+1]
    # Set left bound too min nonzero index if no black space exists
    if left_bound < 0: left_bound = col_nonzero.min()

    # Compute right bound based on black space between indices
    for i in range(len(col_nonzero)-1, 0, -1):
        if col_nonzero[i] < w // 2:
            break
        if col_nonzero[i] - col_nonzero[i-1] > blank_dist:
            right_bound = col_nonzero[i-1]
    # Set right bound too min nonzero index if no black space exists
    if right_bound < 0: right_bound = col_nonzero.max()

    # Compute top bound based on black space between indices
    for i in range(0, len(row_nonzero) - 1):
        if row_nonzero[i] > h // 2:
            break
        if row_nonzero[i+1] - row_nonzero[i] > blank_dist:
            top_bound = row_nonzero[i+1]
    # Set top bound too min nonzero index if no black space exists
    if top_bound < 0: top_bound = row_nonzero.min()

    # Compute bottom bound based on black space between indices
    for i in range(len(row_nonzero)-1, 0, -1):
        if row_nonzero[i] < h // 2:
            break
        if row_nonzero[i] - row_nonzero[i-1] > blank_dist:
            bottom_bound = row_nonzero[i-1]
    # Set bottom bound too min nonzero index if no black space exists
    if bottom_bound < 0: bottom_bound = row_nonzero.max()

    # Store the rough crop indices
    top = int(top_bound)
    bottom = int(bottom_bound) + 1
    left = int(left_bound)
    right = int(right_bound) + 1

    indices = (top, bottom, left, right)

    # Crop the original image based on rough crop indices
    image = image[top:bottom, left:right]

    # Plot the current stage of the image and print crop indices
    if verbose:
        print('Initial Crop:', indices)
        plot_image(image, title='Initial Crop')

    return image, indices

def quadrant_indices(quadrant_array, quad_height, quad_width, quadrant, sum_threshold=0.10):
    """
    For a given quadrant of the image array, return the corresponding
    indices for the boundaries obtained for the second stage of cropping.

    Parameters
    ----------
    quadrant_array : ndarray
        the quadrant of the image
    quad_height : int
        height/#rows of the quadrant
    quad_width : int
        width/#columns of the quadrants
    quadrant : str
        text specifying which quadrant to get indices for
    sum_threshold : float
        percentage of pixels of the row/column desired to
        contain information to be kept within the crop boundaries

    Returns
    -------
    Combination of left/top, left/bottom, right/top, or right/bottom
        indices to adjust the crop on the image
    """
    # Convert array to binary on whether pixels contain info or not
    binary = quadrant_array > 0

    # Calculate row and column sums of binary array and find indices
    # with greather than a threshold number of pixels with data
    col_binary = binary.sum(axis=0) > quad_height * sum_threshold
    row_binary = binary.sum(axis=1) > quad_width * sum_threshold

    # Based on specified quadrant, return relevant boundary indices
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
    elif quadrant == 'topright':
        try:
            right_bound = col_binary.nonzero()[0].max()
        except:
            right_bound = quad_width
        try:
            top_bound = row_binary.nonzero()[0].min()
        except:
            top_bound = 0

        return int(right_bound), int(top_bound)
    elif quadrant == 'bottomleft':
        try:
            left_bound = col_binary.nonzero()[0].min()
        except:
            left_bound = 0
        try:
            bottom_bound = row_binary.nonzero()[0].max()
        except:
            bottom_bound= quad_height

        return int(left_bound), int(bottom_bound)
    elif quadrant == 'bottomright':
        try:
            right_bound = col_binary.nonzero()[0].max()
        except:
            right_bound = quad_width
        try:
            bottom_bound = row_binary.nonzero()[0].max()
        except:
            bottom_bound = quad_height

        return int(right_bound), int(bottom_bound)
    elif quadrant == 'centerleft':
        try:
            left_bound = np.where(col_binary == 0)[0][-1]
            return int(left_bound)
        except:
            return 0
    elif quadrant == 'centerright':
        try:
            right_bound = np.where(col_binary == 0)[0][0]
            return int(right_bound)
        except:
            return quad_width

def get_quad_crop_offsets(image, verbose=False):
    """
    Second stage option of cropping the DICOM image by splitting the
    array into quadrants and looking at empty space within each. After
    indices are found from each quadrant, the tightest bounds are chosen
    and the image is cropped with them plus an additional amount given
    by pixel_crop.

    Parameters
    ----------
    image : ndarray
        array of the DICOM image after initial crop
    verbose : bool
        If True, print out the index offsets using quadrant cropping

    Returns
    -------
    offsets : int, (top, bottom, left, right)
        pixel offsets to constrict after rough crop
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

    # Store the offsets
    offsets = (top_bound, bottom_bound, left_bound, right_bound)

    # Print second stage crop offsets
    if verbose:
        print('Quadrant crop offsets:', offsets)

    return offsets

def get_center_crop_offsets(image, verbose=False):
    """
    Second stage option of cropping the DICOM image by looking at the
    middle 50% region of the image, splitting it into left and right
    sections, and finding empty space within each. After indices are
    found from each quadrant, the tightest bounds are chosen and the
    image is cropped with them plus an additional amount given by
    pixel_crop.

    Parameters
    ----------
    image : ndarray
        array of the DICOM image after initial crop
    verbose : bool
        If True, print out the index offsets using center cropping

    Returns
    -------
    offsets : int, (top, bottom, left, right)
        pixel offsets to constrict after rough crop
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

    # Store the offsets
    offsets = (top_bound, bottom_bound, left_bound, right_bound)

    # Print second stage crop offsets
    if verbose:
        print('Center crop offsets:', offsets)

    return offsets

def unet_crop(image, pixel_spacing, model, verbose=False):
    """
    Second stage option of cropping the DICOM image by using a trained
    U-Net network to find an instance segmentation of the thoracic
    cavity and ribs and finding the smallest and largest indices of
    those masks.

    Parameters
    ----------
    image : ndarray
        array of the DICOM image after initial crop
    pixel_spacing : list [row_spacing, column_spacing]
        row and column spacing of image in physical units
    model : Tensorflow-Keras model
        U-Net architecture model to use segmentation instead
        of region crop for second stage of cropping
    verbose : bool
        If True, print out the index offsets using center cropping

    Returns
    -------
    cat_y_pred : ndarray
        processed, categorical segmentation mask of shape (h, w)
    offsets : int, (top, bottom, left, right)
        pixel offsets to constrict after rough crop
    """
    # Make copy of image to prep for U-Net
    image_copy = image.copy()

    # Resize and normalize to [0, 1] for U-Net
    image_copy = cv2.resize(image_copy, (256, 256), interpolation=cv2.INTER_NEAREST)
    image_copy = (image_copy - np.min(image_copy)) / (np.max(image_copy) - np.min(image_copy))

    # Predict segementation mask with U-Net and check if "failed"
    y_pred = model.predict(image_copy[np.newaxis, ..., np.newaxis]).squeeze(axis=0)  # remove "batch" dimension
    y_pred = to_binary(y_pred)  # threshold to one-hot

    # Convert predicted segmentation to categorical (h, w)
    cat_y_pred = np.argmax(y_pred, axis=-1)

    # Process prediction (keep largest connected component)
    # and check if U-Net has "failed"
    cat_y_pred, unet_failed = postprocess(cat_y_pred, pixel_spacing)

    # Resize prediction to rough crop dimensions to obtain proper offsets (OPENCV NEEDS (W, H) TUPLE FOR RESIZE)
    cat_y_pred = cv2.resize(cat_y_pred, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Return nonsense offsets if U-Net segmentation "failed"
    if unet_failed:
        return cat_y_pred, (-1, -1, -1, -1)

    # Find offsets based on processed, predicted mask
    offsets = get_unet_offsets(cat_y_pred)

    # Print second stage crop offsets
    if verbose:
        print('U-Net segmentation crop offsets:', offsets)

    return cat_y_pred, offsets

def crop_dicom(image, pixel_spacing=None, verbose=False, crop_region='center', model=None):
    """
    Full function to crop a DICOM image. The pixel array is cropped
    through two stages: the first does a rough crop to center on the
    X-Ray, the second eliminates glowing/white borders and as much empty
    space as possible. The offsets are preserved for adjusting
    annotations later.

    Parameters
    ----------
    image : ndarray
        array of the original DICOM image
    pixel_spacing : float
        the physical spacing between pixels from the imaging device
    verbose : bool
        If True, print out each stage of image processing steps
    crop_region : str
        determine between quadrant or center regions to determine
        the second stage of cropping
    model : TensorFlow model
        U-Net architecture model to use instance segmentation instead
        of region crop for second stage of cropping

    Returns
    -------
    y_pred : ndarray
        fully processed and cropped segmentation mask
    indices : int, (top, bottom, left, right)
        the pixel offsets needed to adjust annotations
    """
    # Make a copy of the image to modify
    image_copy = image.copy()
    orig_image_shape = image.shape

    # Threshold the image copy for improved cropping
    image_copy = threshold_image(image_copy, method='li', verbose=verbose)

    # Get the indices for the rough crop stage
    image_copy, init_crop = rough_crop(image_copy, verbose=verbose)

    # Get the index offsets from the region crop stage or U-Net segmentation
    if model is not None:
        cat_y_pred, region_offsets = unet_crop(image_copy, pixel_spacing, model, verbose=verbose)

        # If U-Net failed, revert to non-U-Net region crop
        if region_offsets == (-1, -1, -1, -1):
            if crop_region == 'quad':
                region_offsets = get_quad_crop_offsets(image_copy, verbose=verbose)
            elif crop_region == 'center':
                region_offsets = get_center_crop_offsets(image_copy, verbose=verbose)
    else:
        if crop_region == 'quad':
            region_offsets = get_quad_crop_offsets(image_copy, verbose=verbose)
        elif crop_region == 'center':
            region_offsets = get_center_crop_offsets(image_copy, verbose=verbose)

    # Calculate rough crop + unet/region crop indices
    indices = (max(0, init_crop[0] + region_offsets[0]),
               min(image.shape[0], init_crop[1] - region_offsets[1]),
               max(0, init_crop[2] + region_offsets[2]),
               min(image.shape[1], init_crop[3] - region_offsets[3]))

    # Plot the final cropped image
    if verbose: plot_image(image, title='Final Cropped Image')

    if model is not None:
        # Pad pred array with zeros based on rough crop indices to match original image shape
        cat_y_pred = np.pad(cat_y_pred, [(init_crop[0], orig_image_shape[0] - init_crop[1]),
                                         (init_crop[2], orig_image_shape[1] - init_crop[3])])
        
        # Convert to one-hot (H, W, n_classes)
        y_pred = to_one_hot(cat_y_pred)

        return y_pred, indices
    else:
        return indices

def save_to_npy(y_pred, save_loc):
    """
    Save NumPy array to a .npy file.

    Parameters
    ----------
    y_pred : ndarray
        fully processed, cropped segmentation mask
    save_loc : str
        filepath of destination of image_array
    """
    np.save(save_loc, y_pred)

def create_rgb(image):
    """
    Stack the array to create a 3-channel RGB array.

    Parameters
    ----------
    image : ndarray
        single-channel array of the image

    Returns
    -------
    image : ndarray
        three-channel array of the image
    """
    # Stack the array three times for RGB channels
    image = np.stack([image, image, image], axis=-1)

    return image

def save_to_png(image_array, save_loc, overwrite=False):
    """
    Save the image array to a RGB PNG file.

    Parameters
    ----------
    image_array : ndarray
        the fully processed DICOM image
    save_loc : str
        directory location and filename to save the PNG to
    overwrite : bool
        True/False on whether to overwrite an existing image
    """
    if overwrite:
        numpngw.write_png(save_loc, image_array)
    else:
        if not os.path.exists(save_loc):
            numpngw.write_png(save_loc, image_array)

def scale_image_to_depth(image, bit_depth):
    """
    Takes the input image array and scales the  values to the
    specified bit depth.

    Parameters
    ----------
    image : ndarray
        array of the image
    bit_depth : int
        bit depth the image will be saved as
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

def hist_equalization(image, method='hand', bit_depth=16, verbose=False):
    """
    Perform histogram equalization on the input image, dependent on the
    method chosen. OpenCV requires the image to be 8-bit, and skimage
    outputs an array with values between [0,1].

    Parameters
    ----------
    image : ndarray
        array of the image
    method : str
        the chosen method to perform histogram equalization
        (choices : 'opencv', 'skimage', 'hand')
    bit_depth : int
        max bit depth of the array to equalize to
        (only used if method=='hand')
    verbose : bool
        If True, print out the equalized image

    Returns
    -------
    hist_eq_img : ndarray
        histogram equalized image
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
        L = 2**bit_depth # 65536 for 16, 256 for 8

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

    # Plot the histogram equalized image
    if verbose: plot_image(hist_eq_img, title='Histogram Equalized Image')

    return hist_eq_img

def extract_bboxes(annotation_data):
    """
    Pull out the top left and bottom right (x,y) locations for boundary
    boxes.

    Parameters
    ----------
    annotation_data : dict
        information for the DICOM image annotations

    Returns
    -------
    tl_xs : list
        list of top left boundary box x-values
    tl_ys : list
        list of top left boundary box y-values
    br_xs : list
        list of bottom right boundary box x-values
    br_ys : list
        list of bottom right boundary box y-values
    """
    tl_xs = []
    tl_ys = []
    br_xs = []
    br_ys = []
    for pt in zip(annotation_data['rectangles']['start_points'], annotation_data['rectangles']['end_points']):
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

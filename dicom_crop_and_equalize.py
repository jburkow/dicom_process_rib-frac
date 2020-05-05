'''
Filename: dicom_crop_and_equalize.py
Author: Jonathan Burkow, burkowjo@msu.edu
        Michigan State University
Last Updated: 04/14/2020
Description: Loops through the list of DICOM files
    with image information and crops the image, and
    performs histogram equalization. Versions of the
    original, cropped, and equalized images in both
    8-bit depth and 16-bit depth are saved.
'''

import platform
import numpy as np
from pydicom import dcmread
import os
import time
from dicom_utils import load_dicom_image, crop_dicom, hist_equalization, create_rgb, image_scale_to_8bit, save_to_png

# Set current day string
timestr = time.strftime("%Y%m%d")

# Set arguments
ARGS = {'break' : False,
        'SAVE_FOLDER' : 'fracture_present_1Feb2020_{}'.format(timestr),
        '8_BIT_FOLDER' : '8bit_images',
        '16_BIT_FOLDER' : '16bit_images',
        'ORIGINAL_IMAGE_FOLDER' : 'original_png',
        'CROPPED_IMAGE_FOLDER' : 'cropped_png',
        'EQUALIZED_IMAGE_FOLDER' : 'crop_histeq_png',
        'OFFSET_FILENAME' : 'dicom_offsets.csv'}

# Set path to the data folder
if platform.system() == 'Windows': # Local Laptop directory
    ARGS['DATA_PATH'] = os.path.join('C:\\Users\\JonathanBurkow\\Documents\\Research\\rib_fracture_detection\\', ARGS['SAVE_FOLDER'])
elif platform.system() == 'Linux': # CookieMonster directory
    ARGS['DATA_PATH'] = ''
    
# Make sure data path exists
if not os.path.isdir(ARGS['DATA_PATH']):
    os.mkdir(ARGS['DATA_PATH'])

# Set up 8 bit folder paths
folder_8bit = os.path.join(ARGS['DATA_PATH'], ARGS['8_BIT_FOLDER'])
original_8bit_folder = os.path.join(folder_8bit, ARGS['ORIGINAL_IMAGE_FOLDER'])
cropped_8bit_folder = os.path.join(folder_8bit, ARGS['CROPPED_IMAGE_FOLDER'])
equalized_8bit_folder = os.path.join(folder_8bit, ARGS['EQUALIZED_IMAGE_FOLDER'])

# Set up 16 bit folder paths
folder_16bit = os.path.join(ARGS['DATA_PATH'], ARGS['16_BIT_FOLDER'])
original_16bit_folder = os.path.join(folder_16bit, ARGS['ORIGINAL_IMAGE_FOLDER'])
cropped_16bit_folder = os.path.join(folder_16bit, ARGS['CROPPED_IMAGE_FOLDER'])
equalized_16bit_folder = os.path.join(folder_16bit, ARGS['EQUALIZED_IMAGE_FOLDER'])

# Offset file path -- rows are (IMG, X_OFFSET, Y_OFFSET)
offset_filename = os.path.join(ARGS['DATA_PATH'], ARGS['OFFSET_FILENAME'])
    
# Check for existence of folders
if not os.path.isdir(folder_8bit):
    os.mkdir(folder_8bit)
if not os.path.isdir(original_8bit_folder):
    os.mkdir(original_8bit_folder)
if not os.path.isdir(cropped_8bit_folder):
    os.mkdir(cropped_8bit_folder)
if not os.path.isdir(equalized_8bit_folder):
    os.mkdir(equalized_8bit_folder)
    
if not os.path.isdir(folder_16bit):
    os.mkdir(folder_16bit)
if not os.path.isdir(original_16bit_folder):
    os.mkdir(original_16bit_folder)
if not os.path.isdir(cropped_16bit_folder):
    os.mkdir(cropped_16bit_folder)
if not os.path.isdir(equalized_16bit_folder):
    os.mkdir(equalized_16bit_folder)

# Print out start of execution
print('Starting execution...')
start_time = time.time()

# Import the dataset list
dataset_list = []
with open('dicom_dataset.csv', 'r') as data_file:
    for line in data_file:
        dataset_list.append(line.replace('\n', ''))

# Import the annotated Instance UIDs
instance_uids = []
with open('Annotated_Instance_UIDs.csv', 'r') as data_file:
    for line in data_file:
        instance_uids.append(line.replace('\n', ''))
        
# Check for existence of folders
if not os.path.isdir(original_8bit_folder):
    os.mkdir(original_8bit_folder)
if not os.path.isdir(cropped_8bit_folder):
    os.mkdir(cropped_8bit_folder)
if not os.path.isdir(equalized_8bit_folder):
    os.mkdir(equalized_8bit_folder)
    
if not os.path.isdir(original_16bit_folder):
    os.mkdir(original_16bit_folder)
if not os.path.isdir(cropped_16bit_folder):
    os.mkdir(cropped_16bit_folder)
if not os.path.isdir(equalized_16bit_folder):
    os.mkdir(equalized_16bit_folder)

# Loop through all dicom files and process all annotated instances
offset_list = []
for i, file in enumerate(dataset_list):
    if ARGS['break'] and i == 15:
        break
        
    try:
        print('Processing image {} of {} ({}%).'.format(i+1, len(dataset_list), round((i+1)/len(dataset_list)*100,1)), end='\r')

        # Grab filename
        filename = file[file.rfind('/')+1:file.rfind('_')]

        # Check whether files already exist. If so, continue to next image
        count_8bit = 0
        count_16bit = 0
        for root, dirs, files in os.walk(folder_8bit):
            if filename + '.png' in files:
                count_8bit += 1
        for root, dirs, files in os.walk(folder_16bit):
            if filename + '.png' in files:
                count_16bit += 1

        if count_8bit == 3 and count_16bit == 3:
            continue

        # Load in dicom file
        dcm = dcmread(file)
        
        # Pull out the InstanceUID of the dicom file
        instance_uid = dcm.SOPInstanceUID
        
        # Check if InstanceUID is in the list of annotated instances.
        # If not, continue to next dicom file.
        if instance_uid not in instance_uids:
            continue

        # Retrieve original and cropped image
        original_image = load_dicom_image(dcm)
        cropped_image, offsets = crop_dicom(dcm)

        # Do histogram equalization on the cropped image
        hist_eq_image = hist_equalization(cropped_image, method='skimage')

        # Append x and y index offsets to list

        x_offset = offsets[2]
        y_offset = offsets[0]
        offset_list.append(','.join([filename, str(x_offset), str(y_offset)]))

        # Create 8 bit versions of images
        original_8bit = image_scale_to_8bit(original_image)
        cropped_8bit = image_scale_to_8bit(cropped_image)
        hist_eq_8bit = image_scale_to_8bit(hist_eq_image)

        original_8bit_rgb = create_rgb(original_8bit)
        cropped_8bit_rgb = create_rgb(cropped_8bit)
        hist_eq_8bit_rgb = create_rgb(hist_eq_8bit)

        # Create 16 bit versions of images
        original_16bit = original_image.astype('uint16')
        cropped_16bit = cropped_image.astype('uint16')
        hist_eq_16bit = (hist_eq_image*(2**16 - 1)).astype('uint16')

        original_16bit_rgb = create_rgb(original_16bit)
        cropped_16bit_rgb = create_rgb(cropped_16bit)
        hist_eq_16bit_rgb = create_rgb(hist_eq_16bit)

        # Set the filenames for each image to save to
        original_8bit_path = os.path.join(original_8bit_folder, filename + '.png')
        cropped_8bit_path = os.path.join(cropped_8bit_folder, filename + '.png')
        hist_eq_8bit_path = os.path.join(equalized_8bit_folder, filename + '.png')

        original_16bit_path = os.path.join(original_16bit_folder, filename + '.png')
        cropped_16bit_path = os.path.join(cropped_16bit_folder, filename + '.png')
        hist_eq_16bit_path = os.path.join(equalized_16bit_folder, filename + '.png')

        # Save the images to their respective folders
        save_to_png(original_8bit_rgb, original_8bit_path)
        save_to_png(cropped_8bit_rgb, cropped_8bit_path)
        save_to_png(hist_eq_8bit_rgb, hist_eq_8bit_path)

        save_to_png(original_16bit_rgb, original_16bit_path)
        save_to_png(cropped_16bit_rgb, cropped_16bit_path)
        save_to_png(hist_eq_16bit_rgb, hist_eq_16bit_path)
        
    except:
        print('') # End print stream
        print('Failed on image', filename)

print('') # End print stream from loop

# Export the list of offsets to a file
with open(offset_filename, 'w') as out_file:
    for line in offset_list:
        out_str = line + '\n'
        out_file.write(out_str)

# Print out time to complete
print('Done!')
end_time = time.time()
print('Execution finished in {} seconds.'.format(round(end_time - start_time, 3)))
'''
Filename: dicom_crop_and_equalize.py
Author: Jonathan Burkow, burkowjo@msu.edu
        Michigan State University
Last Updated: 08/10/2020
Description: Loops through the list of DICOM files with image
    information and crops the image, and performs histogram
    equalization. Versions of the original, cropped, and equalized
    images in both 8-bit depth and 16-bit depth are saved.
'''

import csv
import os
import time
import sys
from pydicom import dcmread
from dicom_utils import (load_dicom_image, crop_dicom, hist_equalization, create_rgb,
                         scale_image_to_depth, save_to_png, save_to_npy)
import args
from unet_utils import Unet

# Determine whether to instantiate and load U-Net model based on command line argument
if len(sys.argv) > 1 and sys.argv[1] == 'unet':
    # Instantiate U-Net model and load pre-trained weights
    unet_model = Unet(img_height=256, img_width=256, n_classes=8, n_filters=32)
    unet_model.load_weights(args.ARGS['MODEL_WEIGHTS_FILENAME'])
else:
    unet_model = None

# Print out start of execution
print('Starting execution...')
start_time = time.perf_counter()

# Set up 8 bit folder paths
folder_8bit = args.ARGS['8_BIT_FOLDER']
original_8bit_folder = os.path.join(folder_8bit, args.ARGS['ORIGINAL_IMAGE_FOLDER'])
original_equalized_8bit_folder = os.path.join(folder_8bit, args.ARGS['ORIGINAL_EQUALIZED_IMAGE_FOLDER'])
cropped_8bit_folder = os.path.join(folder_8bit, args.ARGS['CROPPED_IMAGE_FOLDER'])
cropped_equalized_8bit_folder = os.path.join(folder_8bit, args.ARGS['CROPPED_EQUALIZED_IMAGE_FOLDER'])

# Set up 16 bit folder paths
folder_16bit = args.ARGS['16_BIT_FOLDER']
original_16bit_folder = os.path.join(folder_16bit, args.ARGS['ORIGINAL_IMAGE_FOLDER'])
original_equalized_16bit_folder = os.path.join(folder_16bit, args.ARGS['ORIGINAL_EQUALIZED_IMAGE_FOLDER'])
cropped_16bit_folder = os.path.join(folder_16bit, args.ARGS['CROPPED_IMAGE_FOLDER'])
cropped_equalized_16bit_folder = os.path.join(folder_16bit, args.ARGS['CROPPED_EQUALIZED_IMAGE_FOLDER'])

# Set up segmentation mask folder path
folder_seg_mask = args.ARGS['SEG_MASK_FOLDER']
cropped_seg_mask_folder = os.path.join(folder_seg_mask, args.ARGS['CROPPED_MASK_FOLDER'])

# Check for existence of 8-bit folders
if not os.path.isdir(folder_8bit):
    os.mkdir(folder_8bit)
if not os.path.isdir(original_8bit_folder):
    os.mkdir(original_8bit_folder)
if not os.path.isdir(original_equalized_8bit_folder):
    os.mkdir(original_equalized_8bit_folder)
if not os.path.isdir(cropped_8bit_folder):
    os.mkdir(cropped_8bit_folder)
if not os.path.isdir(cropped_equalized_8bit_folder):
    os.mkdir(cropped_equalized_8bit_folder)

# Check for existence of 16-bit folders
if not os.path.isdir(folder_16bit):
    os.mkdir(folder_16bit)
if not os.path.isdir(original_16bit_folder):
    os.mkdir(original_16bit_folder)
if not os.path.isdir(original_equalized_16bit_folder):
    os.mkdir(original_equalized_16bit_folder)
if not os.path.isdir(cropped_16bit_folder):
    os.mkdir(cropped_16bit_folder)
if not os.path.isdir(cropped_equalized_16bit_folder):
    os.mkdir(cropped_equalized_16bit_folder)

# Check for existence of segmentation mask folders
if not os.path.isdir(folder_seg_mask):
    os.mkdir(folder_seg_mask)
if not os.path.isdir(cropped_seg_mask_folder):
    os.mkdir(cropped_seg_mask_folder)

# Import the dataset list
dataset_list = []
with open(args.ARGS['DATASET_LIST'], 'r') as data_file:
    for line in data_file:
        dataset_list.append(line.replace('\n', ''))

# Import the annotated Instance UIDs
instance_uids = []
with open(args.ARGS['INSTANCE_UID_FILENAME'], 'r') as data_file:
    csv_reader = csv.reader(data_file)
    for line in csv_reader:
        instance_uids.append(line[0])

# Loop through all dicom files and process all annotated instances
offset_list = []
failed_list = []
for i, file in enumerate(dataset_list):
    if "Anon_IB_1472" not in file:
        continue
    if args.ARGS['BREAK'] and i == args.ARGS['CROP_BREAK_NUM']:
        break

    try:
        print('Processing image {} of {} ({}%).'.format(i+1, len(dataset_list), round((i+1)/len(dataset_list)*100,1)), end='\r')

        # Grab filename
        filename = file[file.rfind('/')+1:file.rfind('_')]

        # Check whether files already exist in all 4 sub-folders. If so, continue to next image
        count_8bit = 0
        count_16bit = 0
        for root, dirs, files in os.walk(folder_8bit):
            if filename + '.png' in files:
                count_8bit += 1
        for root, dirs, files in os.walk(folder_16bit):
            if filename + '.png' in files:
                count_16bit += 1

        if count_8bit == 4 and count_16bit == 4:
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
        if unet_model is not None:
            cropped_image, pred_mask, offsets = crop_dicom(dcm, model=unet_model)
        else:
            cropped_image, offsets = crop_dicom(dcm, model=unet_model)

        # Do histogram equalization on the cropped image
        original_histeq_image = hist_equalization(original_image, method='skimage')
        cropped_histeq_image = hist_equalization(cropped_image, method='skimage')

        # Append x and y index offsets to list
        x_offset = offsets[2]
        y_offset = offsets[0]
        offset_list.append(','.join([filename, str(x_offset), str(y_offset)]))

        # Create 8 bit versions of images
        original_8bit = scale_image_to_depth(original_image, 8)
        original_histeq_8bit = scale_image_to_depth(original_histeq_image, 8)
        cropped_8bit = scale_image_to_depth(cropped_image, 8)
        cropped_histeq_8bit = scale_image_to_depth(cropped_histeq_image, 8)

        original_8bit_rgb = create_rgb(original_8bit)
        original_histeq_8bit_rgb = create_rgb(original_histeq_8bit)
        cropped_8bit_rgb = create_rgb(cropped_8bit)
        cropped_histeq_8bit_rgb = create_rgb(cropped_histeq_8bit)

        # Create 16 bit versions of images
        original_16bit = scale_image_to_depth(original_image, 16)
        original_histeq_16bit = scale_image_to_depth(original_histeq_image, 16)
        cropped_16bit = scale_image_to_depth(cropped_image, 16)
        cropped_histeq_16bit = scale_image_to_depth(cropped_histeq_image, 16)

        original_16bit_rgb = create_rgb(original_16bit)
        original_histeq_16bit_rgb = create_rgb(original_histeq_16bit)
        cropped_16bit_rgb = create_rgb(cropped_16bit)
        cropped_histeq_16bit_rgb = create_rgb(cropped_histeq_16bit)

        # Set the filenames for each image to save to
        original_8bit_path = os.path.join(original_8bit_folder, filename + '.png')
        original_histeq_8bit_path = os.path.join(original_equalized_8bit_folder, filename + '.png')
        cropped_8bit_path = os.path.join(cropped_8bit_folder, filename + '.png')
        cropped_histeq_8bit_path = os.path.join(cropped_equalized_8bit_folder, filename + '.png')

        original_16bit_path = os.path.join(original_16bit_folder, filename + '.png')
        original_histeq_16bit_path = os.path.join(original_equalized_16bit_folder, filename + '.png')
        cropped_16bit_path = os.path.join(cropped_16bit_folder, filename + '.png')
        cropped_histeq_16bit_path = os.path.join(cropped_equalized_16bit_folder, filename + '.png')

        # Set filename for cropped, processed segmentation mask
        cropped_seg_mask_path = os.path.join(cropped_seg_mask_folder, filename + '.npy')

        # Save the images to their respective folders
        save_to_png(original_8bit_rgb, original_8bit_path)
        save_to_png(original_histeq_8bit_rgb, original_histeq_8bit_path)
        save_to_png(cropped_8bit_rgb, cropped_8bit_path)
        save_to_png(cropped_histeq_8bit_rgb, cropped_histeq_8bit_path)

        save_to_png(original_16bit_rgb, original_16bit_path)
        save_to_png(original_histeq_16bit_rgb, original_histeq_16bit_path)
        save_to_png(cropped_16bit_rgb, cropped_16bit_path)
        save_to_png(cropped_histeq_16bit_rgb, cropped_histeq_16bit_path)

        # Save cropped, processed segmentation mask
        if unet_model is not None:
            save_to_npy(pred_mask, cropped_seg_mask_path)

    except Exception as e:
        print('') # End print stream from loop
        print(e)
        failed_list.append(filename)

print('') # End print stream from loop

# Print out failed-to-process images:
if len(failed_list) > 0:
    print("Failed on", len(failed_list), "images:")
    [print(img) for img in failed_list]

# Export the list of offsets to a file
# Rows are (IMG, X_OFFSET, Y_OFFSET)
if not args.ARGS['BREAK']:
    with open(args.ARGS['OFFSET_FILENAME'], 'w') as out_file:
        for line in offset_list:
            out_str = line + '\n'
            out_file.write(out_str)

# Print out time to complete
print('Done!')
end_time = time.perf_counter()
print('Execution finished in {} seconds.'.format(round(end_time - start_time, 3)))

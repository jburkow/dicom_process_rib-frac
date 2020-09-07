'''
Filename: dicom_crop_and_equalize.py
Author: Jonathan Burkow, burkowjo@msu.edu
        Michigan State University
Last Updated: 09/06/2020
Description: Loops through the list of DICOM files with image
    information and crops the image, and performs histogram
    equalization. Versions of the original, cropped, and equalized
    images in both 8-bit depth and 16-bit depth are saved. CSV files
    with the original and cropped annotation information are also saved.
'''

import csv
import os
import time
import sys
import json
import pandas as pd
from pydicom import dcmread
from dicom_utils import (load_dicom_image, crop_dicom, hist_equalization, create_rgb, create_dir,
                         scale_image_to_depth, save_to_png, save_to_npy, extract_bboxes)
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

# Set up 8-bit directory paths
folder_8bit = args.ARGS['8_BIT_FOLDER']
original_8bit_folder = os.path.join(folder_8bit, args.ARGS['ORIGINAL_IMAGE_FOLDER'])
original_equalized_8bit_folder = os.path.join(folder_8bit, args.ARGS['ORIGINAL_EQUALIZED_IMAGE_FOLDER'])
cropped_8bit_folder = os.path.join(folder_8bit, args.ARGS['CROPPED_IMAGE_FOLDER'])
cropped_equalized_8bit_folder = os.path.join(folder_8bit, args.ARGS['CROPPED_EQUALIZED_IMAGE_FOLDER'])

# Set up 16-bit directory paths
folder_16bit = args.ARGS['16_BIT_FOLDER']
original_16bit_folder = os.path.join(folder_16bit, args.ARGS['ORIGINAL_IMAGE_FOLDER'])
original_equalized_16bit_folder = os.path.join(folder_16bit, args.ARGS['ORIGINAL_EQUALIZED_IMAGE_FOLDER'])
cropped_16bit_folder = os.path.join(folder_16bit, args.ARGS['CROPPED_IMAGE_FOLDER'])
cropped_equalized_16bit_folder = os.path.join(folder_16bit, args.ARGS['CROPPED_EQUALIZED_IMAGE_FOLDER'])

# Set up segmentation mask directory path
folder_seg_mask = args.ARGS['SEG_MASK_FOLDER']
cropped_seg_mask_folder = os.path.join(folder_seg_mask, args.ARGS['CROPPED_MASK_FOLDER'])

# Create directories for 8-bit images
create_dir(folder_8bit)
create_dir(original_8bit_folder)
create_dir(original_equalized_8bit_folder)
create_dir(cropped_8bit_folder)
create_dir(cropped_equalized_8bit_folder)

# Create directories for 16-bit images
create_dir(folder_16bit)
create_dir(original_16bit_folder)
create_dir(original_equalized_16bit_folder)
create_dir(cropped_16bit_folder)
create_dir(cropped_equalized_16bit_folder)

# Create directories for segmentation masks
create_dir(folder_seg_mask)
create_dir(cropped_seg_mask_folder)

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
original_annotations = []
offset_annotations = []
for i, file in enumerate(dataset_list):
    if args.ARGS['BREAK'] and i == args.ARGS['CROP_BREAK_NUM']:
        break

    try:
        print('Processing image {} of {} ({}%).'.format(i+1,
                                                        len(dataset_list),
                                                        round((i+1)/len(dataset_list)*100, 1)),
              end='\r', flush=True)

        # Grab Patient ID
        patient_id = file[file.rfind('/')+1:file.rfind('_')]

        # Pull the corresponding annotation filename from annotation folder
        annotation_filename = [fname for fname in os.listdir(args.ARGS['ANNOTATION_FOLDER']) if patient_id in fname]

        # Pull out annotation information
        with open(os.path.join(args.ARGS['ANNOTATION_FOLDER'], annotation_filename[0])) as json_file:
            annotation_data = json.load(json_file)

        # Extract lists of bounding box points from annotation file
        tl_xs, tl_ys, br_xs, br_ys = extract_bboxes(annotation_data)

        # Check whether files already exist in all 4 sub-folders. If so, continue to next image
        COUNT_8BIT = 0
        COUNT_16BIT = 0
        for root, dirs, files in os.walk(folder_8bit):
            if patient_id + '.png' in files:
                COUNT_8BIT += 1
        for root, dirs, files in os.walk(folder_16bit):
            if patient_id + '.png' in files:
                COUNT_16BIT += 1

        # if COUNT_8BIT == 4 and COUNT_16BIT == 4:
        #     continue

        # Load in dicom file
        dcm = dcmread(file)

        # Pull out the InstanceUID of the dicom file
        instance_uid = dcm.SOPInstanceUID

        # Pull out ImagerPixelSpacing if it is available
        PIXEL_SPACING = dcm.ImagerPixelSpacing if hasattr(dcm, "ImagerPixelSpacing") else None

        # Check if InstanceUID is in the list of annotated instances.
        # If not, continue to next dicom file.
        if instance_uid not in instance_uids:
            continue

        # Load in original image and get indices to crop
        original_image = load_dicom_image(dcm)
        if unet_model is not None:
            pred_mask, offsets = crop_dicom(original_image, pixel_spacing=PIXEL_SPACING, model=unet_model)
        else:
            offsets = crop_dicom(original_image, model=unet_model)

        # Compare crop indices to bounding boxes
        # If bounding boxes are outside of crop indices, use bounding box indices to crop
        minmax_indices = (min(offsets[0], min(tl_ys)),
                          max(offsets[1], max(br_ys)),
                          min(offsets[2], min(tl_xs)),
                          max(offsets[3], max(br_xs)))

        # Use ImagerPixelSpacing if available to add buffer to all sides
        if PIXEL_SPACING is not None:
            row_spacing = int(args.ARGS['MM_SPACING_BUFFER'] / PIXEL_SPACING[0])
            col_spacing = int(args.ARGS['MM_SPACING_BUFFER'] / PIXEL_SPACING[1])
            final_indices = (max(0, minmax_indices[0] - row_spacing),
                             min(original_image.shape[0], minmax_indices[1] + row_spacing),
                             max(0, minmax_indices[2] - col_spacing),
                             min(original_image.shape[1], minmax_indices[3] + col_spacing))
        else:
            final_indices = minmax_indices

        # Crop image
        cropped_image = original_image.copy()[final_indices[0]:final_indices[1], final_indices[2]:final_indices[3]]

        # Do histogram equalization on the cropped image
        original_histeq_image = hist_equalization(original_image, method='skimage')
        cropped_histeq_image = hist_equalization(cropped_image, method='skimage')

        # Append x and y index offsets to list
        x_offset = final_indices[2]
        y_offset = final_indices[0]
        offset_list.append(','.join([patient_id, str(x_offset), str(y_offset)]))

        # Create new lists for bounding boxes integrating the offsets
        offset_tl_xs = [val - x_offset for val in tl_xs]
        offset_tl_ys = [val - y_offset for val in tl_ys]
        offset_br_xs = [val - x_offset for val in br_xs]
        offset_br_ys = [val - y_offset for val in br_ys]

        for x1, y1, x2, y2 in zip(tl_xs, tl_ys, br_xs, br_ys):
            info = [os.path.join(original_8bit_folder, patient_id + '.png'),
                    original_image.shape[0],
                    original_image.shape[1],
                    x1,
                    y1,
                    x2,
                    y2]
            original_annotations.append(info)

        for x1, y1, x2, y2 in zip(offset_tl_xs, offset_tl_ys, offset_br_xs, offset_br_ys):
            info = [os.path.join(cropped_equalized_8bit_folder, patient_id + '.png'),
                    cropped_image.shape[0],
                    cropped_image.shape[1],
                    x1,
                    y1,
                    x2,
                    y2]
            offset_annotations.append(info)

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
        original_8bit_path = os.path.join(original_8bit_folder, patient_id + '.png')
        original_histeq_8bit_path = os.path.join(original_equalized_8bit_folder, patient_id + '.png')
        cropped_8bit_path = os.path.join(cropped_8bit_folder, patient_id + '.png')
        cropped_histeq_8bit_path = os.path.join(cropped_equalized_8bit_folder, patient_id + '.png')

        original_16bit_path = os.path.join(original_16bit_folder, patient_id + '.png')
        original_histeq_16bit_path = os.path.join(original_equalized_16bit_folder, patient_id + '.png')
        cropped_16bit_path = os.path.join(cropped_16bit_folder, patient_id + '.png')
        cropped_histeq_16bit_path = os.path.join(cropped_equalized_16bit_folder, patient_id + '.png')

        # Set filename for cropped, processed segmentation mask
        cropped_seg_mask_path = os.path.join(cropped_seg_mask_folder, patient_id + '.npy')

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
        failed_list.append(patient_id)

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
            OUT_STR = line + '\n'
            out_file.write(OUT_STR)

# Export original and offset annotation lists to files
orig_annotations_df = pd.DataFrame(original_annotations, columns=(['ID', 'height', 'width', 'x1', 'y1', 'x2', 'y2']))
orig_annotations_df.to_csv(args.ARGS['ANNOTATION_OG_FILENAME'], index=False, header=False)

offset_annotations_df = pd.DataFrame(offset_annotations, columns=(['ID', 'height', 'width', 'x1', 'y1', 'x2', 'y2']))
offset_annotations_df.to_csv(args.ARGS['ANNOTATION_OFFSET_FILENAME'], index=False, header=False)

# Print out time to complete
print('Done!')
end_time = time.perf_counter()
print('Execution finished in {} seconds.'.format(round(end_time - start_time, 3)))

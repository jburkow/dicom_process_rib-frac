'''
Filename: dicom_crop_and_equalize.py
Author: Jonathan Burkow, burkowjo@msu.edu
        Michigan State University
Last Updated: 04/22/2021
Description: Loops through the list of DICOM files with image
    information and crops the image, and performs histogram
    equalization. Versions of the original, cropped, and equalized
    images in both 8-bit depth and 16-bit depth are saved. CSV files
    with the original and cropped annotation information are also saved.
'''

import argparse
import os
import time
import json
import traceback
import torch
import pandas as pd
from pydicom import dcmread
from dicom_utils import (load_dicom_image, crop_dicom, hist_equalization, create_rgb,
                         scale_image_to_depth, save_to_png, save_to_npy, extract_bboxes)
from args import ARGS
from ChestSeg_PyTorch.models.UNet_3Plus import UNet_3Plus_DeepSup
from general_utils import read_file, print_iter

def main(parse_args):
    """Main Function"""
    # Import the dataset list
    dataset_list = read_file(ARGS['DATASET_LIST'])

    # Import the annotated Instance UIDs
    instance_uids = read_file(ARGS['INSTANCE_UID_FILENAME'])

    # If --unet was used, load U-Net model
    if parse_args.unet:
        device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
        unet_model = UNet_3Plus_DeepSup(in_channels=1, n_classes=7, n_filters=32).to(device)
        unet_model.load_state_dict(torch.load(parse_args.model_weights)['weights'])
    else:
        unet_model = None

    # Loop through all dicom files and process all annotated instances
    offset_list = []
    failed_list = []
    original_annotations = []
    offset_annotations = []
    for i, file in enumerate(dataset_list):
        # Break after a certain number of images, if desired
        if parse_args.break_loop and i == parse_args.break_num:
            break

        try:
            print_iter(len(dataset_list), i, 'image')

            # Grab Patient ID
            patient_id = file[file.rfind('/')+1:file.rfind('_')]

            # Check whether files already exist in all 4 sub-folders. If so, continue to next image
            if not parse_args.overwrite and not parse_args.just_annos:
                count_8bit = 0
                count_16bit = 0
                for _, _, files in os.walk(ARGS['8_BIT_FOLDER']):
                    if patient_id + '.png' in files:
                        count_8bit += 1
                for _, _, files in os.walk(ARGS['16_BIT_FOLDER']):
                    if patient_id + '.png' in files:
                        count_16bit += 1

                if count_8bit == 4 and count_16bit == 4:
                    continue

            # Load in dicom file
            dcm = dcmread(file)

            # Check if InstanceUID is in the list of annotated instances.
            # If yes, get corresponding annotation info
            if dcm.SOPInstanceUID in instance_uids:
                # Pull the corresponding annotation filename from annotation folder
                annotation_filename = [fname for fname in os.listdir(ARGS['ANNOTATION_FOLDER']) if patient_id in fname]

                # Pull out annotation information
                with open(os.path.join(ARGS['ANNOTATION_FOLDER'], annotation_filename[0])) as json_file:
                    annotation_data = json.load(json_file)

                # Extract lists of bounding box points from annotation file
                tl_xs, tl_ys, br_xs, br_ys = extract_bboxes(annotation_data)

            # Pull out ImagerPixelSpacing if it is available
            PIXEL_SPACING = dcm.ImagerPixelSpacing if hasattr(dcm, "ImagerPixelSpacing") else None

            # Load in original image and get indices to crop
            original_image = load_dicom_image(dcm)

            if unet_model is not None:
                pred_mask, offsets = crop_dicom(original_image, pixel_spacing=PIXEL_SPACING, model=unet_model, device=device)
            else:
                offsets = crop_dicom(original_image, model=unet_model)

            # Compare crop indices to bounding boxes
            # If not in InstanceUIDs, keep crop indices the same
            if dcm.SOPInstanceUID not in instance_uids:
                minmax_indices = (min(offsets[0]),
                                  max(offsets[1]),
                                  min(offsets[2]),
                                  max(offsets[3]))
            # If bounding boxes are outside of crop indices, use bounding box indices to crop
            minmax_indices = (min(offsets[0], min(tl_ys)),
                              max(offsets[1], max(br_ys)),
                              min(offsets[2], min(tl_xs)),
                              max(offsets[3], max(br_xs)))

            # Use ImagerPixelSpacing if available to add buffer to all sides
            if PIXEL_SPACING is not None:
                row_spacing = int(parse_args.mm_spacing / PIXEL_SPACING[0])
                col_spacing = int(parse_args.mm_spacing / PIXEL_SPACING[1])
                final_indices = (max(0, minmax_indices[0] - row_spacing),
                                 min(original_image.shape[0], minmax_indices[1] + row_spacing),
                                 max(0, minmax_indices[2] - col_spacing),
                                 min(original_image.shape[1], minmax_indices[3] + col_spacing))
            else:
                final_indices = minmax_indices

            # Crop image and segmentation mask together
            cropped_image = original_image.copy()[final_indices[0]:final_indices[1], final_indices[2]:final_indices[3]]
            cropped_pred_mask = pred_mask[final_indices[0]:final_indices[1], final_indices[2]:final_indices[3], :]

            # Do histogram equalization on the cropped image
            original_histeq_image = hist_equalization(original_image, method='skimage')
            cropped_histeq_image = hist_equalization(cropped_image, method='skimage')

            # Append x and y index offsets to list
            x_offset = final_indices[2]
            y_offset = final_indices[0]
            offset_list.append(','.join([patient_id, str(x_offset), str(y_offset)]))

            if dcm.SOPInstanceUID in instance_uids:
                # Create new lists for bounding boxes integrating the offsets
                offset_tl_xs = [val - x_offset for val in tl_xs]
                offset_tl_ys = [val - y_offset for val in tl_ys]
                offset_br_xs = [val - x_offset for val in br_xs]
                offset_br_ys = [val - y_offset for val in br_ys]

                # Append annotations based on the original image
                for x1, y1, x2, y2 in zip(tl_xs, tl_ys, br_xs, br_ys):
                    info = [os.path.join(ARGS['8_BIT_OG_IMAGE_FOLDER'], patient_id + '.png'),
                            original_image.shape[0],
                            original_image.shape[1],
                            x1,
                            y1,
                            x2,
                            y2]
                    original_annotations.append(info)

                # Append annotations based on the cropped image
                for x1, y1, x2, y2 in zip(offset_tl_xs, offset_tl_ys, offset_br_xs, offset_br_ys):
                    info = [os.path.join(ARGS['8_BIT_CROP_HISTEQ_IMAGE_FOLDER'], patient_id + '.png'),
                            cropped_image.shape[0],
                            cropped_image.shape[1],
                            x1,
                            y1,
                            x2,
                            y2]
                    offset_annotations.append(info)
            else: # Append images without annotations
                info = [os.path.join(ARGS['8_BIT_OG_IMAGE_FOLDER'], patient_id + '.png'),
                        original_image.shape[0],
                        original_image.shape[1],
                        "", "", "", ""]
                original_annotations.append(info)
                
                info = [os.path.join(ARGS['8_BIT_CROP_HISTEQ_IMAGE_FOLDER'], patient_id + '.png'),
                        cropped_image.shape[0],
                        cropped_image.shape[1],
                        "", "", "", ""]
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
            original_8bit_path = os.path.join(ARGS['8_BIT_OG_IMAGE_FOLDER'], patient_id + '.png')
            original_histeq_8bit_path = os.path.join(ARGS['8_BIT_OG_HISTEQ_IMAGE_FOLDER'], patient_id + '.png')
            cropped_8bit_path = os.path.join(ARGS['8_BIT_CROP_IMAGE_FOLDER'], patient_id + '.png')
            cropped_histeq_8bit_path = os.path.join(ARGS['8_BIT_CROP_HISTEQ_IMAGE_FOLDER'], patient_id + '.png')

            original_16bit_path = os.path.join(ARGS['16_BIT_OG_IMAGE_FOLDER'], patient_id + '.png')
            original_histeq_16bit_path = os.path.join(ARGS['16_BIT_OG_HISTEQ_IMAGE_FOLDER'], patient_id + '.png')
            cropped_16bit_path = os.path.join(ARGS['16_BIT_CROP_IMAGE_FOLDER'], patient_id + '.png')
            cropped_histeq_16bit_path = os.path.join(ARGS['16_BIT_CROP_HISTEQ_IMAGE_FOLDER'], patient_id + '.png')

            # Save the images to their respective folders
            if not parse_args.just_annos:
                save_to_png(original_8bit_rgb, original_8bit_path, overwrite=parse_args.overwrite)
                save_to_png(original_histeq_8bit_rgb, original_histeq_8bit_path, overwrite=parse_args.overwrite)
                save_to_png(cropped_8bit_rgb, cropped_8bit_path, overwrite=parse_args.overwrite)
                save_to_png(cropped_histeq_8bit_rgb, cropped_histeq_8bit_path, overwrite=parse_args.overwrite)

                save_to_png(original_16bit_rgb, original_16bit_path, overwrite=parse_args.overwrite)
                save_to_png(original_histeq_16bit_rgb, original_histeq_16bit_path, overwrite=parse_args.overwrite)
                save_to_png(cropped_16bit_rgb, cropped_16bit_path, overwrite=parse_args.overwrite)
                save_to_png(cropped_histeq_16bit_rgb, cropped_histeq_16bit_path, overwrite=parse_args.overwrite)

            # Set filename for cropped, processed segmentation mask
            original_seg_mask_path = os.path.join(ARGS['ORIGINAL_MASK_FOLDER'], patient_id + '.npy')
            cropped_seg_mask_path = os.path.join(ARGS['CROPPED_MASK_FOLDER'], patient_id + '.npy')

            # Save processed original size and cropped segmentation mask
            if unet_model is not None and not parse_args.just_annos:
                save_to_npy(pred_mask, original_seg_mask_path)
                save_to_npy(cropped_pred_mask, cropped_seg_mask_path)

        except Exception as e:
            print('') # End print stream from loop
            print(traceback.format_exc())
            failed_list.append(patient_id)
    print('') # End print stream from loop

    # Print out failed-to-process images:
    if len(failed_list) > 0:
        print(f"Failed on {len(failed_list)} images:")
        for img in failed_list:
            print(img)

    # Export the list of offsets to a file
    # Rows are (IMG, X_OFFSET, Y_OFFSET)
    if not parse_args.break_loop:
        with open(ARGS['OFFSET_FILENAME'], 'w') as out_file:
            for line in offset_list:
                out_str = line + '\n'
                out_file.write(out_str)

    # Export original and offset annotation lists to files
    orig_annotations_df = pd.DataFrame(original_annotations, columns=(['ID', 'height', 'width', 'x1', 'y1', 'x2', 'y2']))
    offset_annotations_df = pd.DataFrame(offset_annotations, columns=(['ID', 'height', 'width', 'x1', 'y1', 'x2', 'y2']))

    # Save files only when loop isn't being broken (i.e., tested)
    if not parse_args.break_loop:
        orig_annotations_df.to_csv(ARGS['ANNOTATION_OG_FILENAME'], index=False, header=False)
        offset_annotations_df.to_csv(ARGS['ANNOTATION_OFFSET_FILENAME'], index=False, header=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pre-process all images by cropping and equalizing them. Saves images as both 8-bit and 16-bit.')

    parser.add_argument('--unet', action='store_true',
                        help='Utilize a trained U-Net for second-stage cropping.')

    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing images.')

    parser.add_argument('--model_weights', default='ChestSeg_PyTorch/logs/UNet3+_32f_256_bs24_bce+iou_deep-sup_aug_adam-0.0001/chkpt_epoch-256.pt',
                        help='Filename/path to the model weights to load into U-Net.')

    parser.add_argument('--mm_spacing', type=int, default=5,
                        help='Amount of buffer to add around the image post-cropping (in mm).')

    parser.add_argument('--break_loop', action='store_true',
                        help='Condition to break the loop early after break_num iterations.')

    parser.add_argument('--break_num', type=int, default=5,
                        help='Number of iterations to break loop after.')

    parser.add_argument('--just_annos', action='store_true',
                        help='Use if you just need to get annotation and offset files remade.')

    parser_args = parser.parse_args()

    # Print out start of execution
    print('Starting execution...')
    start_time = time.perf_counter()

    # Run main function
    main(parser_args)

    # Print out time to complete
    print('Done!')
    end_time = time.perf_counter()
    print('Execution finished in {} seconds.'.format(round(end_time - start_time, 3)))

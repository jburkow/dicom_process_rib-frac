'''
Filename: check_dimensions.py
Author: Jonathan Burkow, burkowjo@msu.edu
        Michigan State University
Last Updated: 02/04/2021
Description: Loads in the dicom_offsets.csv file, and loops through all
    cropped/equalized images to check whether the annotation offsets
    fall outside the dimensions of the cropped images.
'''

import argparse
import os
import time
import imageio
import pandas as pd
from args import ARGS
from general_utils import print_iter

def main(parse_args):
    """Main Function"""
    # Create list of processed images
    imgs = sorted(os.listdir(ARGS['8_BIT_CROP_HISTEQ_IMAGE_FOLDER']))

    # Load in offset annotation information
    anno_df = pd.read_csv(ARGS['ANNOTATION_OFFSET_FILENAME'], names=['PatientID', 'x1', 'y1', 'x2', 'y2', 'class'])

    # Drop class column
    anno_df = anno_df.drop('class', axis=1)

    # Remove paths from PatientID
    anno_df['PatientID'] = anno_df['PatientID'].str.replace(ARGS['8_BIT_CROP_HISTEQ_IMAGE_FOLDER'] + '/', '')

    # Loop through images and store dimensions in a new DataFrame
    heights = []
    widths = []
    for i, img in enumerate(imgs):
        print_iter(len(imgs), i, 'image')

        img_path = os.path.join(ARGS['8_BIT_CROP_HISTEQ_IMAGE_FOLDER'], img)
        image = imageio.imread(img_path)
        heights.append(image.shape[0])
        widths.append(image.shape[1])

    # Add PatientID, heights, and widths to a new DataFrame
    dim_df = pd.DataFrame({'PatientID' : imgs, 'Height' : heights, 'Width' : widths})

    # Merge the annotation DataFrame with the dimension DataFrame
    merged_df = anno_df.merge(dim_df, on='PatientID')

    # Make new colums based on whether annotation indices are outside dimensions
    merged_df['x1_outside'] = merged_df['x1'] > merged_df['Width']
    merged_df['x2_outside'] = merged_df['x2'] > merged_df['Width']
    merged_df['y1_outside'] = merged_df['y1'] > merged_df['Height']
    merged_df['y2_outside'] = merged_df['y2'] > merged_df['Height']

    # Save merged_df to a CSV file
    print('Writing to file...')
    merged_df.to_csv(parse_args.filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Checks all offset annotations to make sure they fall within the dimensions of cropped images.')

    parser.add_argument('--filename', type=str, default='dimension_check.csv',
                        help='Filename to save to.')

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

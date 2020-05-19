'''
Filename: create_dicom_dataset_list.py
Author: Jonathan Burkow, burkowjo@msu.edu
        Michigan State University
Last Updated: 05/12/2020
Description: Goes through the provided dataset location of DICOM
    files and creates a file listing all which contain PixelData
    information.
'''

import platform
import os
from pydicom import dcmread
import time
import args

# Print out start of execution
print('Starting execution...')
start_time = time.time()

# Set paths for original DICOM folder and CSV
dicom_folder = os.path.join(args.ARGS['BASE_DATA_PATH'], args.ARGS['DICOM_FOLDER'])
csv_path = os.path.join(args.ARGS['BASE_DATA_PATH'], args.ARGS['PROCESSED_SAVE_FOLDER'], 'dicom_dataset.csv')

# Check if the processed save folder exists
if not os.path.isdir(os.path.join(args.ARGS['BASE_DATA_PATH'], args.ARGS['PROCESSED_SAVE_FOLDER'])):
    os.mkdir(os.path.join(args.ARGS['BASE_DATA_PATH'], args.ARGS['PROCESSED_SAVE_FOLDER']))

full_dataset_list = []
has_image_dataset_list = []
no_has_image_dataset_list = []
for dirs, root, files in os.walk(dicom_folder):
    for file in files:
        full_dataset_list.append(os.path.join(dirs,file))

for item in full_dataset_list:
    dcm = dcmread(item)
    if hasattr(dcm, 'PixelData'):
        has_image_dataset_list.append(item)
    else:
        no_has_image_dataset_list.append(item)
        
# Print out number of DICOM files and how many have pixel information
print('Total number of DICOM files:', len(full_dataset_list))
print('DICOM files containing pixel data:', len(has_image_dataset_list))
print('DICOM files without pixel data:', len(no_has_image_dataset_list))

# Save the list of paths to DICOM files with pixel information to a file
with open(csv_path, 'w') as out_file:
    for row in has_image_dataset_list:
        out_str = row + '\n'
        out_file.write(out_str)
        
# Print out time to complete
print('Done!')
end_time = time.time()
print('Execution finished in {} seconds.'.format(round(end_time - start_time, 3)))
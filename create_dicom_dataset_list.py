'''
Filename: create_dicom_dataset_list.py
Author: Jonathan Burkow, burkowjo@msu.edu
        Michigan State University
Last Updated: 04/10/2020
Description: Goes through the provided dataset location of DICOM
    files and creates a file listing all which contain PixelData
    information.
'''

import platform
import os
from pydicom import dcmread
import time

ARGS = {}

if platform.system() == 'Windows': # Local Laptop directory
    ARGS['DATA_PATH'] = 'C:\\Users\\JonathanBurkow\\Documents\\Research\\rib_fracture_detection\\fracture_present_1Feb2020'
elif platform.system() == 'Linux': # CookieMonster directory
    ARGS['DATA_PATH'] = '/data/mididata/rib_fracture_id/fracture_present_1Feb2020'

# Print out start of execution
print('Starting execution...')
start_time = time.time()

full_dataset_list = []
has_image_dataset_list = []
no_has_image_dataset_list = []
for dirs, root, files in os.walk(ARGS['DATA_PATH']):
    for file in files:
        full_dataset_list.append(dirs[:-1] + file)

for item in full_dataset_list:
    dcm = dcmread(item)
    if hasattr(dcm, 'PixelData'):
        has_image_dataset_list.append(item)
    else:
        no_has_image_dataset_list.append(item)
        
# Print out number of DICOM files and how many have pixel information
print('Total number of DICOM files:', len(full_dataset_list))
print('DICOM files containing pixel data:', len(has_image_dataset_list))
print('DICOM files without pixel data:')
# for image in no_has_image_dataset_list:
#     print(image)

# Save the list of paths to DICOM files with pixel information to a file
with open('dicom_dataset.csv', 'w') as out_file:
    for row in has_image_dataset_list:
        out_str = row + '\n'
        out_file.write(out_str)
        
# Print out time to complete
print('Done!')
end_time = time.time()
print('Execution finished in {} seconds.'.format(round(end_time - start_time, 3)))
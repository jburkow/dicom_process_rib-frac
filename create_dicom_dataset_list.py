'''
Filename: create_dicom_dataset_list.py
Author: Jonathan Burkow, burkowjo@msu.edu
        Michigan State University
Last Updated: 01/27/2021
Description: Goes through the provided dataset location of DICOM files
    and creates a file listing all which have annotations.

    Note: This likely does not need to be run until more data
    is added as the annotated DICOMs should remain the same
    throughout multiple reads.
'''

import os
import csv
import time
from pydicom import dcmread
import args
from general_utils import print_iter

def main():
    """Main Function"""
    # Load in InstanceUIDs
    instance_uids = []
    with open(args.ARGS['INSTANCE_UID_FILENAME'], 'r') as data_file:
        csv_reader = csv.reader(data_file)
        for line in csv_reader:
            instance_uids.append(line[0])

    # Create a list of all DICOM files
    full_dataset_list = []
    for dirs, _, files in os.walk(args.ARGS['DICOM_FOLDER']):
        for file in files:
            full_dataset_list.append(os.path.join(dirs, file))

    # Create list of annotated DICOM files
    annotated_dicoms = []
    for i, dcm_file in enumerate(full_dataset_list):
        print_iter(len(full_dataset_list), i, 'DICOM')

        # Load in DICOM file
        dcm = dcmread(dcm_file)

        # Extract InstanceUID from header data
        dcm_instance_uid = dcm.SOPInstanceUID

        # If InstanceUID matches an annotated UID, store it
        if dcm_instance_uid in instance_uids:
            annotated_dicoms.append(dcm_file)
    print('') # End print stream from loop

    # Print out number of DICOM files that have annotations
    print('Annotated DICOM files:', len(annotated_dicoms))

    # Save the list of paths to DICOM files with annotations to a file
    print('Writing to file...')
    with open(args.ARGS['DATASET_LIST'], 'w+') as out_file:
        for row in annotated_dicoms:
            out_str = row + '\n'
            out_file.write(out_str)


if __name__ == "__main__":
    # Print out start of execution
    print('Starting execution...')
    start_time = time.perf_counter()

    # Run main function
    main()

    # Print out time to complete
    print('Done!')
    end_time = time.perf_counter()
    print('Execution finished in {} seconds.'.format(round(end_time - start_time, 3)))

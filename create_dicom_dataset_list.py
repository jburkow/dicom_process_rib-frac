'''
Filename: create_dicom_dataset_list.py
Author: Jonathan Burkow, burkowjo@msu.edu
        Michigan State University
Last Updated: 01/18/2021
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
    for dcm_file in os.listdir(args.ARGS['DICOM_FOLDER']):
        file_path = os.path.join(args.ARGS['DICOM_FOLDER'], dcm_file)
        dcm = dcmread(file_path)

        dcm_instance_uid = dcm.SOPInstanceUID

        if dcm_instance_uid in instance_uids:
            annotated_dicoms.append(file_path)

    # Print out number of DICOM files and how many have pixel information
    print('Total number of DICOM files:', len(full_dataset_list))
    print('Annotated DICOM files:', len(annotated_dicoms))

    # Save the list of paths to DICOM files with pixel information to a file
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

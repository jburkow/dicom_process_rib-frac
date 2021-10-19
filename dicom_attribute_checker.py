'''
Filename: dicom_attribute_checker.py
Author: Jonathan Burkow (burkowjo@msu.edu), Michigan State University
Last Updated: 10/19/2021
Description: Collect all DICOM datasets together (ANON_IB_1xxx, fracture_unknown, and ANON_IB_2xxx)
    and choose a DICOM attribute to search and print out all unique values.
'''

import os
import time
from typing import List
from tqdm import tqdm
import pydicom


def get_unique_list(dicom_list: List[str]):
    unique_list = []
    for dicom in dicom_list:
        dicom_file = dicom.split('/')[-1]
        patient_id = dicom_file[:dicom_file.rfind('_')]
        unique_list.append(patient_id)

    return set(unique_list)


def main():
    frac_pres_dir = '/mnt/home/burkowjo/midi_lab/burkowjo_data/fracture_present_1Feb2020'
    frac_abs_dir = '/mnt/home/burkowjo/midi_lab/matched_fractureabsent_19Nov2020'
    frac_pres2_dir = '/mnt/home/burkowjo/midi_lab/burkowjo_data/fracture_present_8June2021'

    frac_pres_list = sorted([os.path.join(root, file) for root, _, files in os.walk(frac_pres_dir) for file in files])
    frac_pres2_list = sorted([os.path.join(root, file) for root, _, files in os.walk(frac_pres2_dir) for file in files])
    frac_abs_list = sorted([os.path.join(root, file) for root, _, files in os.walk(frac_abs_dir) for file in files])

    all_dicoms = frac_pres_list + frac_pres2_list + frac_abs_list

    print('                       Total | Unique')
    print(f'All DICOM files       : {len(all_dicoms)} | {len(get_unique_list(all_dicoms))}')
    print(f'ANON_IB_1xxx files    :  {len(frac_pres_list)} | {len(get_unique_list(frac_pres_list))}')
    print(f'fracture_unknown files:  {len(frac_abs_list)} | {len(get_unique_list(frac_abs_list))}')
    print(f'ANON_IB_2xxx files    :  {len(frac_pres2_list)} | {len(get_unique_list(frac_pres2_list))}')

    all_attrs = []

    # attribute = 'PhotometricInterpretation'
    attribute = 'BodyPartExamined'

    for _, dicom in tqdm(enumerate(all_dicoms), desc=f'Getting unique {attribute}', total=len(all_dicoms)):
        dcm = pydicom.dcmread(dicom)

        tmp_attr = dcm.BodyPartExamined if hasattr(dcm, attribute) else 'NA'

        all_attrs.append(tmp_attr)

    unique_interps = set(all_attrs)

    print(f'Unique {attribute}')
    for attr in unique_interps:
        print(attr)


if __name__ == "__main__":
    # Print out start of execution
    print('\nStarting execution...')
    start_time = time.perf_counter()

    # Run main function
    main()

    # Print out time to complete
    print('Done!')
    end_time = time.perf_counter()
    print(f'Execution finished in {end_time - start_time:.3f} seconds.\n')

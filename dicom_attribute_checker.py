'''
Filename: dicom_attribute_checker.py
Author: Jonathan Burkow (burkowjo@msu.edu), Michigan State University
Last Updated: 10/26/2021
Description: Collect all DICOM datasets together (ANON_IB_1xxx, fracture_unknown, and ANON_IB_2xxx)
    and choose a DICOM attribute to search and print out all unique values.
'''

import os
import time
from typing import List
from tqdm import tqdm
import pydicom
from tabulate import tabulate


def get_unique_list(dicom_list: List[str]) -> List[str]:
    """
    Return a list of all unique patients from the given dataset.

    Parameters
    ----------
    dicom_list : list of DICOM images with their associated paths.
    """
    unique_list = []
    for dicom in dicom_list:
        dicom_file = dicom.split('/')[-1]
        patient_id = dicom_file[:dicom_file.rfind('_')]
        unique_list.append(patient_id)
    return list(set(unique_list))


def retrieve_all_attributes(dicom_list: List[str], attribute) -> List[str]:
    """
    Loop through all DICOM files in dicom_list and return a list of all unique attributes found.

    Parameters
    ----------
    dicom_list : list of DICOM images with their associated paths.
    """
    all_attrs = []
    for _, dicom in tqdm(enumerate(dicom_list), desc=f'Getting all unique attributes - {attribute}', total=len(dicom_list)):
        dcm = pydicom.dcmread(dicom)
        tmp_attr = dcm.BodyPartExamined if hasattr(dcm, attribute) else 'NA'
        all_attrs.append(tmp_attr)
    return list(set(all_attrs))


def main():
    frac_pres_dir = '/mnt/home/burkowjo/midi_lab/burkowjo_data/fracture_present_1Feb2020'
    frac_abs_dir = '/mnt/home/burkowjo/midi_lab/matched_fractureabsent_19Nov2020'
    frac_pres2_dir = '/mnt/home/burkowjo/midi_lab/burkowjo_data/fracture_present_8June2021'

    frac_pres_list = sorted([os.path.join(root, file) for root, _, files in os.walk(frac_pres_dir) for file in files])
    frac_pres2_list = sorted([os.path.join(root, file) for root, _, files in os.walk(frac_pres2_dir) for file in files])
    frac_abs_list = sorted([os.path.join(root, file) for root, _, files in os.walk(frac_abs_dir) for file in files])

    all_dicoms = frac_pres_list + frac_pres2_list + frac_abs_list

    # Print out total number of DICOM images and unique patients for each dataset
    print_table = [
        ['', 'TOTAL', 'UNIQUE'],
        ['All DICOM files', len(all_dicoms), len(get_unique_list(all_dicoms))],
        ['ANON_IB_1xxx files', len(frac_pres_list), len(get_unique_list(frac_pres_list))],
        ['fracture_unknown files', len(frac_abs_list), len(get_unique_list(frac_abs_list))],
        ['ANON_IB_2xxx files', len(frac_pres2_list), len(get_unique_list(frac_pres2_list))]
    ]
    print(tabulate(print_table, headers='firstrow', tablefmt="fancy_grid", numalign='center'))
    print()


    # attribute = 'PhotometricInterpretation'
    attribute = 'BodyPartExamined'

    unique_attrs = retrieve_all_attributes(all_dicoms, attribute)

    print(f'Unique {attribute}')
    for attr in unique_attrs:
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

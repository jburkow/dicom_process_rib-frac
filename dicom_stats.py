'''
Filename: dicom_stats.py
Author: Jonathan Burkow, burkowjo@msu.edu
        Michigan State University
Last Updated: 05/07/2021
Description: Loops through all available DICOM files to extract gender,
    age, and other potentially relevant information.
'''

import argparse
import os
import time
from pydicom import dcmread
from args import ARGS
from general_utils import print_iter
import numpy as np


def age_to_days(age_val):
    age_code = age_val[-1]
    tmp_age = 0

    if age_code == 'D':
        tmp_age = int(age_val[:-1])
    elif age_code == 'W':
        tmp_age = int(age_val[:-1]) * 7
    elif age_code == 'M':
        tmp_age = int(age_val[:-1]) * 7 * 4
    elif age_code == 'Y':
        tmp_age = int(age_val[:-1]) * 365
    else:
        raise ValueError("Wtf age code is that?", age_code)

    return tmp_age

def main(parse_args):
    """Main Function"""
    ignore_list = ['0006', '0010', '0071', '0080', '0088', '0099', '0101', '0117', '0139', '0161', '0170']

    frac_present_list = sorted([os.path.join(root, file) for root, _, files in os.walk(ARGS['DICOM_FOLDER']) for file in files])
    unique_frac_present_list = {img[:img.rfind('_')]+'_0.dcm' for img in frac_present_list}
    frac_absent_list = [os.path.join(root, file) for root, _, files in os.walk(ARGS['ABSENT_DICOM_FOLDER']) for file in files if int(file[18:21]) < 201 and file[17:21] not in ignore_list]
    unique_frac_absent_list = {img[:img.rfind('_')]+'_1.dcm' for img in frac_absent_list}

    present_ages = []
    present_genders = np.array([])
    pixel_spacings = np.array([])

    for i, image in enumerate(unique_frac_present_list):
        if parse_args.break_loop and i == parse_args.break_num:
            break
        print_iter(len(unique_frac_present_list), i, 'Fracture Present DICOM')

        dcm = dcmread(image)

        if 'PatientAge' in dcm:
            present_ages.append(dcm['PatientAge'].value)

        if 'PatientSex' in dcm:
            present_genders = np.append(present_genders, 1 if dcm['PatientSex'].value == 'M' else 0)

        if 'PixelSpacing' in dcm:
            pixel_spacings = np.append(pixel_spacings, dcm['PixelSpacing'].value[0])
    print()

    new_present_ages = np.array([])
    for age in present_ages:
        if age == '':
            continue
        new_age = age_to_days(age)
        new_present_ages = np.append(new_present_ages, new_age)
    present_ages_years = new_present_ages / 365.0



    print()

    print(f'{"FRACTURE PRESENT DATA":^{parse_args.width}}')
    print('-'*parse_args.width)

    print('Total DICOM Images:', len(frac_present_list))
    print('Total Unique Patients:', len(unique_frac_present_list))

    print('-'*parse_args.width)

    print(f'{"Age Summary":^{parse_args.width}}')
    print(f'{f"({len(present_ages)}/{len(unique_frac_present_list)}: {len(unique_frac_present_list)-len(present_ages)} missing)":^{parse_args.width}}')
    print()
    print(f'\t avg +/- std. dev. [min - max] [median, Q1, Q3, IQR]')
    print(f'Days: {new_present_ages.mean():^.2f} +/- {new_present_ages.std():.2f} ' \
          f'[{new_present_ages.min()} - {new_present_ages.max()}] ' \
          f'[{np.median(new_present_ages):.1f}, ' \
          f'{np.percentile(new_present_ages, 25)}, {np.percentile(new_present_ages, 75)}, {np.percentile(new_present_ages, 75) - np.percentile(new_present_ages, 25)}]')
    print(f'Years: {present_ages_years.mean():^.3f} +/- {present_ages_years.std():.3f} ' \
          f'[{present_ages_years.min():.3f} - {present_ages_years.max():.3f}] ' \
          f'[{np.median(present_ages_years):.3f}, ' \
          f'{np.percentile(present_ages_years, 25):.3f}, {np.percentile(present_ages_years, 75):.3f}, {np.percentile(present_ages_years, 75) - np.percentile(present_ages_years, 25):.3f}]')

    print('-'*parse_args.width)

    print(f'{"Gender Summary":^{parse_args.width}}')
    print(f'{f"({len(present_genders)}/{len(unique_frac_present_list)}: {len(unique_frac_present_list)-len(present_genders)} missing)":^{parse_args.width}}')
    print()
    print(f'Male: {present_genders.sum():.0f} ({present_genders.sum()/len(present_genders):.1%})')
    print(f'Female: {len(present_genders)-present_genders.sum():.0f} ({(len(present_genders)-present_genders.sum())/len(present_genders):.1%})')

    print('-'*parse_args.width)

    print(f'{"Pixel Spacing Summary":^{parse_args.width}}')
    print(f'{f"({len(pixel_spacings)}/{len(unique_frac_present_list)}: {len(unique_frac_present_list)-len(pixel_spacings)} missing)":^{parse_args.width}}')
    print()
    print(f'\tavg +/- std. dev. [min - max], median')
    print(f'mm: {pixel_spacings.mean():^.3f} +/- {pixel_spacings.std():.3f} [{pixel_spacings.min():.3f} - {pixel_spacings.max():.3f}], {np.median(pixel_spacings):.3f}')

    print('-'*parse_args.width)
    print()






    present_ages = []
    present_genders = np.array([])
    pixel_spacings = np.array([])

    for i, image in enumerate(unique_frac_absent_list):
        if parse_args.break_loop and i == parse_args.break_num:
            break
        print_iter(len(unique_frac_absent_list), i, 'Fracture Present DICOM')

        dcm = dcmread(image)

        if 'PatientAge' in dcm:
            present_ages.append(dcm['PatientAge'].value)

        if 'PatientSex' in dcm:
            present_genders = np.append(present_genders, 1 if dcm['PatientSex'].value == 'M' else 0)

        if 'PixelSpacing' in dcm:
            pixel_spacings = np.append(pixel_spacings, dcm['PixelSpacing'].value[0])
    print()

    new_present_ages = np.array([])
    for age in present_ages:
        if age == '':
            continue
        new_age = age_to_days(age)
        new_present_ages = np.append(new_present_ages, new_age)
    present_ages_years = new_present_ages / 365.0



    print()

    print(f'{"FRACTURE ABSENT DATA":^{parse_args.width}}')
    print('-'*parse_args.width)

    print('Total DICOM Images:', len(frac_absent_list))
    print('Total Unique Patients:', len(unique_frac_absent_list))

    print('-'*parse_args.width)

    print(f'{"Age Summary":^{parse_args.width}}')
    print(f'{f"({len(present_ages)}/{len(unique_frac_absent_list)}: {len(unique_frac_absent_list)-len(present_ages)} missing)":^{parse_args.width}}')
    print()
    print(f'\t avg +/- std. dev. [min - max] [median, Q1, Q3, IQR]')
    print(f'Days: {new_present_ages.mean():^.2f} +/- {new_present_ages.std():.2f} ' \
          f'[{new_present_ages.min():.0f} - {new_present_ages.max():.0f}] ' \
          f'[{np.median(new_present_ages):.1f}, ' \
          f'{np.percentile(new_present_ages, 25):.0f}, {np.percentile(new_present_ages, 75):.0f}, {np.percentile(new_present_ages, 75) - np.percentile(new_present_ages, 25):.0f}]')
    print(f'Years: {present_ages_years.mean():^.3f} +/- {present_ages_years.std():.3f} ' \
          f'[{present_ages_years.min():.3f} - {present_ages_years.max():.3f}] ' \
          f'[{np.median(present_ages_years):.1f}, ' \
          f'{np.percentile(present_ages_years, 25):.3f}, {np.percentile(present_ages_years, 75):.3f}, {np.percentile(present_ages_years, 75) - np.percentile(present_ages_years, 25):.3f}]')

    print('-'*parse_args.width)

    print(f'{"Gender Summary":^{parse_args.width}}')
    print(f'{f"({len(present_genders)}/{len(unique_frac_absent_list)}: {len(unique_frac_absent_list)-len(present_genders)} missing)":^{parse_args.width}}')
    print()
    print(f'Male: {present_genders.sum():.0f} ({present_genders.sum()/len(present_genders):.1%})')
    print(f'Female: {len(present_genders)-present_genders.sum():.0f} ({(len(present_genders)-present_genders.sum())/len(present_genders):.1%})')

    print('-'*parse_args.width)

    print(f'{"Pixel Spacing Summary":^{parse_args.width}}')
    print(f'{f"({len(pixel_spacings)}/{len(unique_frac_absent_list)}: {len(unique_frac_absent_list)-len(pixel_spacings)} missing)":^{parse_args.width}}')
    print()
    print(f'\tavg +/- std. dev. [min - max], median')
    print(f'mm: {pixel_spacings.mean():.3f} +/- {pixel_spacings.std():.3f} [{pixel_spacings.min():.3f} - {pixel_spacings.max():.3f}], {np.median(pixel_spacings):.3f}')

    print('-'*parse_args.width)
    print()




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Print out patient statistics of the DICOM dataset.')

    parser.add_argument('--break_loop', action='store_true',
                        help='Break the loop.')

    parser.add_argument('--break_num', type=int, default=5,
                        help='Iteration to break the loop.')

    parser.add_argument('--width', type=int, default=50,
                        help='Max width of printouts.')


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

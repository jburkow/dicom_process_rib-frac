'''
Filename: dicom_stats.py
Author: Jonathan Burkow, burkowjo@msu.edu
        Michigan State University
Last Updated: 05/10/2021
Description: Loops through all available DICOM files to extract gender,
    age, and other potentially relevant information.
'''

import argparse
import os
import time

from tqdm import tqdm
import numpy as np
import pandas as pd
from pydicom import dcmread

from args import ARGS


def age_to_days(age_val):
    """
    Convert the DICOM-coded age value into a numerical age in days.

    Parameters
    ----------
    age_val : str
        DICOM-coded age representation
    """

    if age_val == '':
        return pd.NA
    if age_val[-1] == 'D':
        return int(age_val[:-1])
    if age_val[-1] == 'W':
        return int(age_val[:-1]) * 7
    if age_val[-1] == 'M':
        return int(age_val[:-1]) * 7 * 4
    if age_val[-1] == 'Y':
        return int(age_val[:-1]) * 365


def simple_manufacturer(manufacturer):
    """
    Simplify the manufacturer name from the DICOM file.

    Parameters
    ----------
    manufacturer : str
        Manufacturer of the X-Ray machine used in the study
    """
    if 'philips' in manufacturer.lower():
        return 'Philips'
    if 'canon' in manufacturer.lower():
        return 'Canon'
    if 'fuji' in manufacturer.lower():
        return 'Fuji'
    if 'kodak' in manufacturer.lower():
        return 'Kodak'
    if 'ge' in manufacturer.lower():
        return 'GE'
    # If none of the previous conditions are met, return "Other"
    return 'Other'

def print_summary(dicom_df, dataset=None):
    """
    Print out summary of metadata from the saved CSV file.

    Parameters
    ----------
    dicom_df : DataFrame
        DataFrame from the loaded in CSV file
    dataset : str
        String to determine whether to print just fracture present or
        fracture absent sections of the data
    """
    if dataset is not None:
        header_str = f'FRACTURE {dataset} SUMMARY'.upper()
        curr_df = dicom_df[dicom_df['has_fractures'] == dataset]
    else:
        header_str = 'FRACTURE DATASET SUMMARY'
        curr_df = dicom_df

    print(f"{header_str:^50}")
    print('-'*50)
    print('Total Unique Patients\t', len(curr_df))
    print('-'*50)
    print(f'{"AGE SUMMARY":^50}')
    print(f'{f"({curr_df.age_days.dropna().shape[0]}/{len(curr_df.age_days)}: {len(curr_df.age_days)-curr_df.age_days.dropna().shape[0]} missing)":^50}')
    print(f'\t{"avg":^8} +/- {"stddev":^8} [{"min":^8}-{"max":^8}] [{"median":^8}, {"q1":^8}, {"q3":^8}, {"iqr":^8}]')
    print(f"Days\t{curr_df.age_days.mean():^8.2f} +/- {curr_df.age_days.std():^8.2f} [{curr_df.age_days.min():^8.2f}-{curr_df.age_days.max():^8.2f}] [{curr_df.age_days.median():^8.2f}, {curr_df.age_days.quantile(0.25):^8.2f}, {curr_df.age_days.quantile(0.75):^8.2f}, {curr_df.age_days.quantile(0.75) - curr_df.age_days.quantile(0.25):^8.2f}]")
    print(f"Years\t{curr_df.age_years.mean():^8.3f} +/- {curr_df.age_years.std():^8.3f} [{curr_df.age_years.min():^8.3f}-{curr_df.age_years.max():^8.3f}] [{curr_df.age_years.median():^8.3f}, {curr_df.age_years.quantile(0.25):^8.3f}, {curr_df.age_years.quantile(0.75):^8.3f}, {curr_df.age_years.quantile(0.75) - curr_df.age_years.quantile(0.25):^8.3f}]")
    print('-'*50)
    print(f'{"GENDER SUMMARY":^50}')
    print(f'{f"({curr_df.gender.dropna().shape[0]}/{len(curr_df.gender)}: {len(curr_df.gender)-curr_df.gender.dropna().shape[0]} missing)":^50}')
    print(f"Male\t{curr_df.male.sum():>5.0f} ({curr_df.male.sum() / len(curr_df):.1%})")
    print(f"Female\t{curr_df.female.sum():>5.0f} ({curr_df.female.sum() / len(curr_df):.1%})")
    print('-'*50)
    print(f'{"PIXEL SPACING SUMMARY":^50}')
    print(f'{f"({curr_df.pixel_spacing.dropna().shape[0]}/{len(curr_df.pixel_spacing)}: {len(curr_df.pixel_spacing)-curr_df.pixel_spacing.dropna().shape[0]} missing)":^50}')
    print(f'\t{"avg":^5} +/- {"stddev":^5} [{"min":^5}-{"max":^5}] [{"median":^5}, {"q1":^5}, {"q3":^5}, {"iqr":^5}]')
    print(f"mm\t{curr_df.pixel_spacing.mean():^5.3f} +/- {curr_df.pixel_spacing.std():^5.3f}  [{curr_df.pixel_spacing.min():^5.3f}-{curr_df.pixel_spacing.max():^5.3f}] [{curr_df.pixel_spacing.median():>6.3f}, {curr_df.pixel_spacing.quantile(0.25):^5.3f}, {curr_df.pixel_spacing.quantile(0.75):^5.3f}, {curr_df.pixel_spacing.quantile(0.75) - curr_df.pixel_spacing.quantile(0.25):^5.3f}]")
    print('-'*50)
    print(f'{"MANUFACTURER SUMMARY":^50}')
    print(f'{f"({curr_df.manufacturer.dropna().shape[0]}/{len(curr_df.manufacturer)}: {len(curr_df.manufacturer)-curr_df.manufacturer.dropna().shape[0]} missing)":^50}')
    print(f"Philips\t{curr_df[curr_df.manufacturer == 'Philips'].shape[0]:>5.0f} ({curr_df[curr_df.manufacturer == 'Philips'].shape[0] / len(curr_df):>5.1%})")
    print(f"Canon\t{curr_df[curr_df.manufacturer == 'Canon'].shape[0]:>5.0f} ({curr_df[curr_df.manufacturer == 'Canon'].shape[0] / len(curr_df):>5.1%})")
    print(f"Fuji\t{curr_df[curr_df.manufacturer == 'Fuji'].shape[0]:>5.0f} ({curr_df[curr_df.manufacturer == 'Fuji'].shape[0] / len(curr_df):>5.1%})")
    print(f"Kodak\t{curr_df[curr_df.manufacturer == 'Kodak'].shape[0]:>5.0f} ({curr_df[curr_df.manufacturer == 'Kodak'].shape[0] / len(curr_df):>5.1%})")
    print(f"GE\t{curr_df[curr_df.manufacturer == 'GE'].shape[0]:>5.0f} ({curr_df[curr_df.manufacturer == 'GE'].shape[0] / len(curr_df):>5.1%})")
    print(f"Other\t{curr_df[curr_df.manufacturer == 'Other'].shape[0]:>5.0f} ({curr_df[curr_df.manufacturer == 'Other'].shape[0] / len(curr_df):>5.1%})")
    print()


def extract_metadata(save_dir):
    """
    Go through all DICOM files and extract relevant metadata, and save to a CSV file.

    Parameters
    ----------
    save_dir : str
        Path to save dicom_metadata.csv to
    """
    # List of images from fracture absent dataset marked as containing 1+ fracture
    ignore_list = ['0006', '0010', '0071', '0080', '0088', '0099', '0101', '0117', '0139', '0161', '0170']

    # Create lists of rib fracture datasets
    frac_present_list = sorted([os.path.join(root, file) for root, _, files in os.walk(ARGS['DICOM_FOLDER']) for file in files])
    unique_frac_present_list = {img[:img.rfind('_')]+'_0.dcm' for img in frac_present_list}
    frac_absent_list = [os.path.join(root, file) for root, _, files in os.walk(ARGS['ABSENT_DICOM_FOLDER']) for file in files if int(file[18:21]) < 201 and file[17:21] not in ignore_list]
    unique_frac_absent_list = {img[:img.rfind('_')]+'_1.dcm' for img in frac_absent_list}
    combined = list(unique_frac_present_list) + list(unique_frac_absent_list)

    # Initialize DataFrame
    dicom_df = pd.DataFrame({'patient_id': [],
                                'has_fractures': [],
                                'age_days': [],
                                'gender': [],
                                'male': [],
                                'female': [],
                                'pixel_spacing': [],
                                'manufacturer': []})

    pbar = tqdm(enumerate(combined), total=len(combined), desc='Processing DICOMs')
    for i, image in pbar:

        dcm = dcmread(image)

        temp_age = age_to_days(dcm['PatientAge'].value) if 'PatientAge' in dcm else pd.NA
        temp_frac = 'present' if 'anon_ib' in image.lower() else 'absent'
        temp_gender = dcm['PatientSex'].value if 'PatientSex' in dcm else ''
        temp_male = 1 if temp_gender == 'M' else 0
        temp_female = 1 if temp_gender == 'F' else 0
        temp_spacing = dcm['PixelSpacing'].value[0] if 'PixelSpacing' in dcm else pd.NA
        temp_manu = simple_manufacturer(dcm['Manufacturer'].value) if 'Manufacturer' in dcm else ''


        dicom_df = dicom_df.append({'patient_id': image,
                                    'has_fractures': temp_frac,
                                    'age_days': temp_age,
                                    'gender': temp_gender,
                                    'male': temp_male,
                                    'female': temp_female,
                                    'pixel_spacing': temp_spacing, 
                                    'manufacturer': temp_manu}, ignore_index=True)

    # Add a column for age in years
    dicom_df.insert(3, 'age_years', dicom_df['age_days'] / 365.0)

    # Save DataFrame to file
    print('Writing to file...')
    dicom_df.to_csv(os.path.join(save_dir, 'dicom_metadata.csv'), index=False)


def main(parse_args):
    """Main Function"""

    if parse_args.extract:
        extract_metadata(parse_args.save_dir)

    if parse_args.summary:
        dicom_df = pd.read_csv(os.path.join(parse_args.save_dir, 'dicom_metadata.csv'))
        print_summary(dicom_df, dataset=parse_args.fractures)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Print out patient statistics of the DICOM dataset.')

    parser.add_argument('--extract', action='store_true',
                        help='Extract metadata from the DICOM files.')

    parser.add_argument('--summary', action='store_true',
                        help='Print out summary of the stored CSV data.')

    parser.add_argument('--break_loop', action='store_true',
                        help='Break the loop.')

    parser.add_argument('--break_num', type=int, default=5,
                        help='Iteration to break the loop.')

    parser.add_argument('--fractures', type=str, default=None,
                        help='Determine what section of data is used for summary (choose one of present or absent).')

    parser.add_argument('--save_dir', default=ARGS['PROCESSED_DATA_FOLDER'],
                        help='Directory to save metadata CSV (default is newest processed folder).')

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

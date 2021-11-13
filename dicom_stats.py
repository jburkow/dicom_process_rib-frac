'''
Filename: dicom_stats.py
Author: Jonathan Burkow (burkowjo@msu.edu), Michigan State University
Last Updated: 11/12/2021
Description: Loop through all DICOM files in dataset to extract gender, age, and other metadata.
'''

import argparse
import os
import time

from tqdm import tqdm
import pandas as pd
from pydicom import dcmread

from args import ARGS
from general_utils import read_file


def age_to_days(age_val: str):
    """
    Convert the DICOM-coded age value into a numerical age in days.

    Parameters
    ----------
    age_val : DICOM-coded age representation
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


def simple_manufacturer(manufacturer: str):
    """
    Simplify the manufacturer name from the DICOM file.

    Parameters
    ----------
    manufacturer : Manufacturer of the X-Ray machine used for the study
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


def print_summary(dicom_df: pd.DataFrame, dataset: str = None):
    """
    Print out summary of metadata from the saved CSV file.

    Parameters
    ----------
    dicom_df : DataFrame from the loaded in CSV file
    dataset  : determine whether to print just fracture present or absent sections of the data;
               choices are 'present' or 'absent'
    """
    if dataset is not None:
        header_str = f'FRACTURE {dataset} SUMMARY'.upper()
        curr_df = dicom_df[dicom_df['has_fractures'] == dataset]
    else:
        header_str = 'FRACTURE DATASET SUMMARY'
        curr_df = dicom_df

    curr_df['manufacturer'] = curr_df['vendor'].apply(lambda x: simple_manufacturer(x))

    print()
    print(f"{header_str:^50}")
    print('='*50)
    print('Total Unique Patients\t', len(curr_df))
    print('='*50)
    print(f'{"AGE SUMMARY":^50}')
    print(f'{f"({curr_df.age_days.dropna().shape[0]}/{len(curr_df.age_days)}: {len(curr_df.age_days)-curr_df.age_days.dropna().shape[0]} missing)":^50}')
    print(f'\t{"avg":^8} +/- {"stddev":^8} [{"min":^8}-{"max":^8}] [{"median":^8}, {"q1":^8}, {"q3":^8}, {"iqr":^8}]')
    print(f"Days\t{curr_df.age_days.mean():^8.2f} +/- {curr_df.age_days.std():^8.2f} [{curr_df.age_days.min():^8.2f}-{curr_df.age_days.max():^8.2f}] [{curr_df.age_days.median():^8.2f}, {curr_df.age_days.quantile(0.25):^8.2f}, {curr_df.age_days.quantile(0.75):^8.2f}, {curr_df.age_days.quantile(0.75) - curr_df.age_days.quantile(0.25):^8.2f}]")
    print(f"Years\t{curr_df.age_years.mean():^8.3f} +/- {curr_df.age_years.std():^8.3f} [{curr_df.age_years.min():^8.3f}-{curr_df.age_years.max():^8.3f}] [{curr_df.age_years.median():^8.3f}, {curr_df.age_years.quantile(0.25):^8.3f}, {curr_df.age_years.quantile(0.75):^8.3f}, {curr_df.age_years.quantile(0.75) - curr_df.age_years.quantile(0.25):^8.3f}]")
    print('='*50)
    print(f'{"GENDER SUMMARY":^50}')
    print(f'{f"({curr_df.gender.dropna().shape[0]}/{len(curr_df.gender)}: {len(curr_df.gender)-curr_df.gender.dropna().shape[0]} missing)":^50}')
    print(f"Male\t{curr_df.male.sum():>5.0f} ({curr_df.male.sum() / len(curr_df):.1%})")
    print(f"Female\t{curr_df.female.sum():>5.0f} ({curr_df.female.sum() / len(curr_df):.1%})")
    print('='*50)
    print(f'{"PIXEL SPACING SUMMARY":^50}')
    print(f'{f"({curr_df.pixel_spacing.dropna().shape[0]}/{len(curr_df.pixel_spacing)}: {len(curr_df.pixel_spacing)-curr_df.pixel_spacing.dropna().shape[0]} missing)":^50}')
    print(f'\t{"avg":^5} +/- {"stddev":^5} [{"min":^5}-{"max":^5}] [{"median":^5}, {"q1":^5}, {"q3":^5}, {"iqr":^5}]')
    print(f"mm\t{curr_df.pixel_spacing.mean():^5.3f} +/- {curr_df.pixel_spacing.std():^5.3f}  [{curr_df.pixel_spacing.min():^5.3f}-{curr_df.pixel_spacing.max():^5.3f}] [{curr_df.pixel_spacing.median():>6.3f}, {curr_df.pixel_spacing.quantile(0.25):^5.3f}, {curr_df.pixel_spacing.quantile(0.75):^5.3f}, {curr_df.pixel_spacing.quantile(0.75) - curr_df.pixel_spacing.quantile(0.25):^5.3f}]")
    print('='*50)
    print(f'{"MANUFACTURER SUMMARY":^50}')
    print(f'{f"({curr_df.manufacturer.dropna().shape[0]}/{len(curr_df.manufacturer)}: {len(curr_df.manufacturer)-curr_df.manufacturer.dropna().shape[0]} missing)":^50}')
    print(f"Philips\t{curr_df[curr_df.manufacturer == 'Philips'].shape[0]:>5.0f} ({curr_df[curr_df.manufacturer == 'Philips'].shape[0] / len(curr_df):>5.1%})")
    print(f"Canon\t{curr_df[curr_df.manufacturer == 'Canon'].shape[0]:>5.0f} ({curr_df[curr_df.manufacturer == 'Canon'].shape[0] / len(curr_df):>5.1%})")
    print(f"Fuji\t{curr_df[curr_df.manufacturer == 'Fuji'].shape[0]:>5.0f} ({curr_df[curr_df.manufacturer == 'Fuji'].shape[0] / len(curr_df):>5.1%})")
    print(f"Kodak\t{curr_df[curr_df.manufacturer == 'Kodak'].shape[0]:>5.0f} ({curr_df[curr_df.manufacturer == 'Kodak'].shape[0] / len(curr_df):>5.1%})")
    print(f"GE\t{curr_df[curr_df.manufacturer == 'GE'].shape[0]:>5.0f} ({curr_df[curr_df.manufacturer == 'GE'].shape[0] / len(curr_df):>5.1%})")
    print(f"Other\t{curr_df[curr_df.manufacturer == 'Other'].shape[0]:>5.0f} ({curr_df[curr_df.manufacturer == 'Other'].shape[0] / len(curr_df):>5.1%})")
    print(f"Unique Vendors: {curr_df.vendor.unique()}")
    print('='*50)
    print()


def extract_metadata(save_dir: str):
    """
    Go through all DICOM files and extract relevant metadata, and save to a CSV file.

    Parameters
    ----------
    save_dir : Path to save dicom_metadata.csv to
    """
    # List of images from the fracture_unknown dataset that contain fractures
    unknown_with_frac_list = ['0006', '0010', '0071', '0080', '0088', '0099', '0101', '0117',
                              '0139', '0161', '0170', '0202', '0225', '0236', '0246', '0250',
                              '0252', '0272', '0296', '0303', '0313', '0317', '0383', '0409',
                              '0426', '0445', '0446', '0450', '0453', '0468']

    # Create list of all DICOM files based on dicom_dataset.csv file
    all_dicoms = read_file(ARGS['DATASET_LIST'])

    # Initialize DataFrame
    dicom_df = pd.DataFrame({'patient_id': [],
                             'has_fractures': [],
                             'age_days': [],
                             'gender': [],
                             'male': [],
                             'female': [],
                             'pixel_spacing': [],
                             'vendor': []})

    for _, image in tqdm(enumerate(all_dicoms), total=len(all_dicoms), desc='Processing DICOMs'):
        dcm = dcmread(image)

        temp_age = age_to_days(dcm['PatientAge'].value) if 'PatientAge' in dcm else pd.NA
        temp_frac = 'present' if 'anon_ib' in image.lower() else 'absent'
        if 'anon_ib' in image.lower():
            temp_frac = 'present'
        else:
            last_underscore_ind = image.rfind('_')
            image_num = image[last_underscore_ind-4:last_underscore_ind]
            temp_frac = 'present' if image_num in unknown_with_frac_list else 'absent'
        temp_gender = dcm['PatientSex'].value if 'PatientSex' in dcm else ''
        temp_male = 1 if temp_gender == 'M' else 0
        temp_female = 1 if temp_gender == 'F' else 0
        temp_spacing = dcm['PixelSpacing'].value[0] if 'PixelSpacing' in dcm else pd.NA
        temp_manu = dcm['Manufacturer'].value if 'Manufacturer' in dcm else ''


        dicom_df = dicom_df.append({'patient_id': image,
                                    'has_fractures': temp_frac,
                                    'age_days': temp_age,
                                    'gender': temp_gender,
                                    'male': temp_male,
                                    'female': temp_female,
                                    'pixel_spacing': temp_spacing, 
                                    'vendor': temp_manu}, ignore_index=True)

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
    parser = argparse.ArgumentParser(description='Print out patient statistics of DICOM dataset.')

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
    print('\nStarting execution...')
    start_time = time.perf_counter()

    # Run main function
    main(parser_args)

    # Print out time to complete
    print('Done!')
    end_time = time.perf_counter()
    print(f'Execution finished in {end_time - start_time:.3f} seconds.\n')

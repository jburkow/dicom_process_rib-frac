'''
Filename: dicom_protocol_descriptions.py
Author: Jonathan Burkow (burkowjo@msu.edu), Michigan State University
Last Updated: 10/27/2021
Description: Goes through a given list of DICOM files and retrieves multiple attribute header
    information to filter out non-AP, non-CHEST images.
'''

import os
import time
from typing import List, Set
import pydicom
from tqdm import tqdm


def get_unique_descriptions(image_list: List[str], attribute: str) -> Set[str]:
    """
    Return a unique set of all descriptions from the image list for the given DICOM attribute.

    Parameters
    ----------
    image_list : list of DICOM images
    attribute  : DICOM attribute to pull information from
    """
    desc_list = []

    for image in image_list:
        dcm = pydicom.dcmread(image)

        if attribute not in dcm:
            continue

        desc_list.append(dcm[attribute].value.upper())

    return set(desc_list)


def get_attribute(dcm: pydicom.Dataset, attribute: str) -> str:
    """
    Returns a concatenation of all information from the attribute about the region and view of the
    DICOM image.

    Parameters
    ----------
    dcm       : DICOM image
    attribute : DICOM attribute to pull information from

    Returns
    -------
    attribute_text : string describing the region and view of the DICOM image
    """
    if attribute not in dcm or dcm[attribute].value == '':
        return ''

    attribute_text = ''

    # if isinstance(dcm[attribute], pydicom.multival.MultiValue):
    #     dicom_text = dcm[attribute][0].upper()
    # else:
    #     dicom_text = dcm[attribute].value.upper()

    try:
        dicom_text = dcm[attribute].value.upper()
    except AttributeError:
        dicom_text = ''

    # Add region information if available
    if 'CHEST' in dicom_text:
        attribute_text += 'CHEST'
    if 'RIB' in dicom_text:
        attribute_text += 'RIB'
    if 'SPINE' in dicom_text:
        attribute_text += 'SPINE'
    if 'ABD' in dicom_text:
        attribute_text += 'ABDOMEN'

    attribute_text += '-' if attribute_text != '' else ''

    # Add view information if available
    if 'AP' in dicom_text:
        attribute_text += 'AP'
    if 'PA' in dicom_text:
        attribute_text += 'PA'
    if 'LAT' in dicom_text or 'LL' in dicom_text:
        attribute_text += 'LAT'

    return attribute_text

def get_body_part(dcm: pydicom.Dataset) -> str:
    """
    Returns the BodyPartExamined information from the DICOM header, if it exists.

    Parameters
    ----------
    dcm : DICOM image
    """
    if 'BodyPartExamined' not in dcm or dcm['BodyPartExamined'].value == '':
        return ''
    try:
        dicom_text = dcm['BodyPartExamined'].value.upper()
    except AttributeError:
        dicom_text = ''
    return dicom_text


def get_all_descriptions(image_list: List[str]) -> List[str]:
    """
    Return a list of simplified descriptions for each image from the image list.

    Parameters
    ----------
    image_list : list of DICOM images
    """
    desc_list = []

    for _, image in tqdm(enumerate(image_list), desc='Pulling Descriptions', total=len(image_list)):
        dcm = pydicom.dcmread(image)

        protocol = get_attribute(dcm, 'ProtocolName')
        seriesdesc = get_attribute(dcm, 'SeriesDescription')
        viewpos = get_attribute(dcm, 'ViewPosition')
        imcomm = get_attribute(dcm, 'ImageComments')
        acqdesc = get_attribute(dcm, 'AcquisitionDeviceProcessingDescription')
        body_part = get_body_part(dcm)
        instanceuid = dcm.SOPInstanceUID

        combined_desc = ','.join([body_part, protocol, seriesdesc, viewpos, imcomm, acqdesc])
        if 'LAT' in combined_desc:
            summary = 'LAT'
        elif 'AP' in combined_desc:
            summary = 'AP'
        elif 'PA' in combined_desc:
            summary = 'PA'
        else:
            summary = 'NA'

        desc_list.append(','.join([instanceuid, summary, combined_desc]))

    return desc_list


def main():
    """Main Function"""
    timestr = time.strftime("%Y%m%d")

    # data_str = 'frac_absent'
    # dataset_dir = '/mnt/home/burkowjo/midi_lab/matched_fractureabsent_19Nov2020'

    data_str = 'frac_present2'
    dataset_dir = '/mnt/home/burkowjo/midi_lab/burkowjo_data/fracture_present_8June2021'

    im_list = sorted([os.path.join(root, file) for root, _, files in os.walk(dataset_dir) for file in files])

    # list_output = get_unique_descriptions(im_list, 'AcquisitionDeviceProcessingDescription')


    list_output = get_all_descriptions(im_list)
    assert len(im_list) == len(list_output)

    with open(f'{timestr}_{data_str}_protocol_list_full.csv', 'w') as out_file:
        for im, line in zip(im_list, list_output):
            out_file.write(im + ',' + line + '\n')

    with open(f'{timestr}_{data_str}_protocol_list_AP.csv', 'w') as out_file:
        for im, line in zip(im_list, list_output):
            if line.split(',')[0] != 'LAT' and ('CHEST' in line or 'AB' in line or 'RIB' in line):
                out_file.write(im + ',' + line + '\n')
                # out_file.write(im + '\n')

    with open(f'{timestr}_{data_str}_protocol_list_CHEST.csv', 'w') as out_file:
        for im, line in zip(im_list, list_output):
            if 'LAT' not in line and ('CHEST' in line.split(',')[2] or
                                      'ABD' in line.split(',')[2] or
                                      'RIB' in line.split(',')[2]):
                out_file.write(im + ',' + line + '\n')
                # out_file.write(im + '\n')


if __name__ == '__main__':

    # Print out start of execution
    print('\nStarting execution...')
    start_time = time.perf_counter()

    # Run main function
    main()

    # Print out time to complete
    print('Done!')
    end_time = time.perf_counter()
    print(f'Execution finished in {end_time - start_time:.3f} seconds.\n')

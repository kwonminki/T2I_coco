from pathlib import Path
import pandas as pd
import shutil
import os
import argparse
import subprocess

#wget http://images.cocodataset.org/zips/val2014.zip
#unzip val2014.zip -d coco/

def download_coco(coco_path="coco", del_zip=False):
    # #wget http://images.cocodataset.org/zips/val2014.zip
    # #unzip val2014.zip -d coco/'
    if not os.path.exists('val2014.zip'):
        coco_download_command = 'wget http://images.cocodataset.org/zips/val2014.zip'
        print(f"Downloading COCO dataset with command: {coco_download_command}")
        process = subprocess.run(coco_download_command, shell=True, check=True)

    if not os.path.exists(coco_path):  
        coco_unzip_command = f'unzip val2014.zip -d {coco_path}/'
        print(f"Unzipping COCO dataset with command: {coco_unzip_command}")
        process = subprocess.run(coco_unzip_command, shell=True, check=True)

    if del_zip:
        # delete zip file
        os.remove('val2014.zip')
    

def split_coco(coco_path):
    df_sample = pd.read_parquet('parquet/subset_coco.parquet')
    subset_path = os.path.join(coco_path, 'subset')
    os.makedirs(subset_path, exist_ok=True)
    for i, row in df_sample.iterrows():
        # caption = row['caption']
        path = 'coco/val2014/' + row['file_name']
        shutil.copy(path, 'coco/subset/') 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='COCO Sampling')
    parser.add_argument('--coco_path', type=str, default='coco', help='Path to COCO dataset')

    args = parser.parse_args()
    if not os.path.exists(args.coco_path):
        download_coco()

    split_coco(args.coco_path)
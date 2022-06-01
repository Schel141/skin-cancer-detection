import pandas as pd
from google.cloud import storage
import cv2
from skin_cancer_detection.hairremoval import hair_removal
import numpy as np
import urllib


AWS_BUCKET_PATH = "s3://wagon-public-datasets/taxi-fare-train.csv"



def get_data_from_gcp(nrows= 5, local=False, optimize=False, **kwargs):
    """method to get the training data (or a portion of it) from google cloud bucket"""
    # Add Client() here
    client = storage.Client()
    if local:
        path = "..raw_data"
    else:
        path = "gs://wagon-data-871-daun/data/HAM10000_all"

    # create df for metadata
    skin_df = pd.read_csv('gs://wagon-data-871-daun/data/HAM10000_metadata.csv', nrows = nrows)

    # load images into df
    skin_df['path'] = [f'gs://wagon-data-871-daun/data/HAM10000_all/{img}.jpg' for img in skin_df['image_id']]

    # hair removal and resizing
    features_list=[]
    for index,row in skin_df.iterrows():
        path = row['path']

        # maybe remove things below again
        resp = urllib.urlopen(path)
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        # image = cv2.imread(path)
        final_image = hair_removal(image)
        image_resize = cv2.resize(final_image,(100,75))
        final_image = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)
        features_list.append(final_image)

    skin_df['image_resized'] = features_list

    return skin_df


def get_data(nrows=10_000):
    '''returns a DataFrame with nrows from s3 bucket'''
    df = pd.read_csv(AWS_BUCKET_PATH, nrows=nrows)
    return df


def clean_data(df, test=False):
    df = df.dropna(how='any', axis='rows')
    df = df[(df.dropoff_latitude != 0) | (df.dropoff_longitude != 0)]
    df = df[(df.pickup_latitude != 0) | (df.pickup_longitude != 0)]
    if "fare_amount" in list(df):
        df = df[df.fare_amount.between(0, 4000)]
    df = df[df.passenger_count < 8]
    df = df[df.passenger_count >= 0]
    df = df[df["pickup_latitude"].between(left=40, right=42)]
    df = df[df["pickup_longitude"].between(left=-74.3, right=-72.9)]
    df = df[df["dropoff_latitude"].between(left=40, right=42)]
    df = df[df["dropoff_longitude"].between(left=-74, right=-72.9)]
    return df


if __name__ == '__main__':
    df = get_data_from_gcp()

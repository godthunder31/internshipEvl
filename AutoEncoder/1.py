import os
import cv2
from sqlalchemy import create_engine, Table, Column, LargeBinary, Integer, MetaData, insert
import pickle
import numpy as np
from scipy.spatial import KDTree

def list_of_image_paths(folder_path):
    image_list = []
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.webp', '.PNG', '.JPG', '.JPEG', '.WEBP')):
            image_list.append(os.path.join(folder_path, filename))
    return image_list

def extract_descriptors(image):
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return descriptors

def binarized_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    height, width = img.shape[:2]
    resized_img = cv2.resize(img, (int(0.25 * width), int(0.25 * height)))
    equalized_image = cv2.equalizeHist(resized_img)
    _, bin_image = cv2.threshold(equalized_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bin_image

folder_path = '/Users/macbook/Oxford/Oxford'
image_list = list_of_image_paths(folder_path)

engine = create_engine("sqlite:///hello.sqlite")
connection = engine.connect()
metadata = MetaData()

sample = Table('a', metadata,
               Column('id', Integer, primary_key=True),
               Column('Image_data', LargeBinary),
               Column('Descriptors', LargeBinary))

kd_tree_table = Table('kd_tree', metadata,
                      Column('id', Integer, primary_key=True),
                      Column('KDTree', LargeBinary))

metadata.create_all(engine)
metadata.reflect(bind=engine)

if 'a' in metadata.tables:
    a1 = metadata.tables['a']
    op = insert(a1)
    records = []
    all_descriptors = []

    for index, convertor in enumerate(image_list):
        word_splitter = os.path.splitext(convertor)[1].lower()
        img = cv2.imread(convertor, cv2.IMREAD_GRAYSCALE)

        descriptors = extract_descriptors(img)
        if descriptors is not None:
            all_descriptors.append(descriptors)

            descriptors_bytes = pickle.dumps(descriptors)

            _, image_bin_for = cv2.imencode(word_splitter, img)
            image_bytes = image_bin_for.tobytes()
            temp1 = {'id': index, 'Image_data': image_bytes, 'Descriptors': descriptors_bytes}
            records.append(temp1)

    if records:
        connection.execute(op, records)
        print("Rows Inserted")

    if all_descriptors:
        all_descriptors = np.vstack(all_descriptors)

        kd_tree = KDTree(all_descriptors)

        kd_tree_bytes = pickle.dumps(kd_tree)

        op = insert(kd_tree_table)
        connection.execute(op, [{'id': 0, 'KDTree': kd_tree_bytes}])
        print("KD-Tree Inserted")

else:
    print("There is no such table")

connection.commit()
connection.close()
import os
import cv2
from sqlalchemy import create_engine, text, MetaData
import pickle
import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt

def extract_keypoints_and_descriptors(image):
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image,None)
    return keypoints, descriptors

engine = create_engine("sqlite:///hello.sqlite")
connection = engine.connect()
metadata = MetaData()

metadata.reflect(bind=engine)

if 'a' in metadata.tables:
    a_table = metadata.tables['a']
    query_a = text('select * from a')
    result_a = connection.execute(query_a).fetchall()

if 'kd_tree' in metadata.tables:
    kd_tree_table = metadata.tables['kd_tree']
    query_kd_tree = text('select * from kd_tree')
    result_kd_tree = connection.execute(query_kd_tree).fetchall()

desc_list = []
index_list = []
for index, samp in enumerate(result_a):
    if samp[2] is not None:
        x = pickle.loads(samp[2])
        desc_list.append(x)
        index_list.extend([index] * x.shape[0])
    else:
        print(f"Descriptors not found for image: {samp[0]}")

kd_tree = pickle.loads(result_kd_tree[0][1])

query_image_path = '/Users/macbook/Oxford/Oxford64/all_souls_000159.jpg'
query_image = cv2.imread(query_image_path,cv2.IMREAD_GRAYSCALE)

keypoints, query_descriptors = extract_keypoints_and_descriptors(query_image)

distances, indices = kd_tree.query(query_descriptors, k=2)

displayed_indices = set()
for i, (m, n) in enumerate(distances):
    if m < 0.5 * n:
        db_index = index_list[indices[i][0]]
        if db_index not in displayed_indices:
            displayed_indices.add(db_index)

            z = np.frombuffer(result_a[db_index][1], np.uint8)
            matched_img = cv2.imdecode(z, cv2.IMREAD_COLOR)
            print("Matched Image Index:", db_index)
            
            matched_img_rgb = cv2.cvtColor(matched_img, cv2.COLOR_BGR2RGB)
            
            plt.figure()
            plt.imshow(matched_img_rgb)
            plt.title(f"Matched Image Index: {db_index}")
            plt.axis('off')
            plt.show()

connection.close()
import numpy as np
import argparse
import cv2
import os
import uuid
import cv2
import mediapipe as mp
import pickle
import argparse
import uuid
import argparse

def create_csv(input,output):
#     path_DTS = "./Real_and_Fake_Face_Detection/"
    path_DTS = input
    # For static images:

    data = []
    label = []

    i = 0

    for folder in os.listdir(path_DTS):
        for file in os.listdir(os.path.join(path_DTS,folder)):
            print("folder :"+str(folder)+"| file :"+file+" |class: ",i)
            img_file = os.path.join(path_DTS,folder,file)
            image = cv2.imread(img_file)

            annotated_image = cv2.resize(image, (128, 128))
            data.append(annotated_image)
            label.append(i)
        i = i + 1

    data = np.array(data)/255
    label = np.array(label)#.reshape(-1,1)

    # from sklearn.preprocessing import LabelBinarizer
    # encoder = LabelBinarizer()
    # label = encoder.fit_transform(label)
    # print(label)

#     file = open('./RAF+ff_ds.csv', 'wb')
    file = open(output, 'wb')
    # dump information to that file
    pickle.dump((data,label), file)
    # close the file
    file.close()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset_dir", required=True,
                    help="path to input dataset")
    ap.add_argument("-o", "--output_dir", required=True,
                    help="output path dataset" default="output.csv")
    args = vars(ap.parse_args())

    create_csv(args.dataset_dir,args.output_dir)
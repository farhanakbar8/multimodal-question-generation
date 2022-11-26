import os
import re
import cv2
import numpy as np
from nltk.corpus import stopwords

def clean_text(text):
    # stop = stopwords.words('english')
    # temp = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", text)
    # return " ".join([word for word in temp.split() if word not in (stop)]).lower().split()
    return [text]

def convert_image(image, dim):
    if image == 'none': return np.zeros((dim,47))
    for img in os.listdir("final_dataset_images"):
        if img.split('.')[0] == image:
            return cv2.resize(cv2.imread(os.path.join("final_dataset_images", img), 0), dsize=(47,dim)) / 255

def dummy_acoustic(dim):
    return np.zeros((dim,74))
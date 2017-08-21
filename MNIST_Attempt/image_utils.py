import numpy as np
import os
import struct
import sys
# from scipy.misc import toimage


# Follow this link to find more info regarding the format of the file:
# http://yann.lecun.com/exdb/mnist/


def load_image_file(filename):
    with open(os.path.join("data", filename), "rb") as f:
        data = f.read()
    f.close()
    
    string_rep = "B" * (len(data)-16)
    string_rep = ">IIII"+ string_rep
    image_data = struct.unpack(string_rep, data)[4:] # Drop the first four elements

    total_image_num = int(len(image_data) / (28*28))
    print("Total images in file: {}".format(total_image_num))
    
    # reshape to 2d np.array where each row is an image of 784 pixels
    im_np_arr = np.asarray(image_data).reshape(total_image_num, 28*28)
    
    return im_np_arr

def load_labels_file(filename):
    with open(os.path.join("data", filename), "rb") as f:
        data = f.read()
    f.close()
    
    string_rep = "B" * (len(data)-8)
    string_rep = ">II"+ string_rep
    label_data = struct.unpack(string_rep, data)[2:] # Drop the first two elements
    
    total_labels = int(len(label_data))
    label_np_arr = np.asarray(label_data).reshape(total_labels, 1)
    
    print("Total labels in file: {}".format(total_labels))
    
    return label_np_arr
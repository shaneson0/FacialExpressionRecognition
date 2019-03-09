import numpy as np
import cv2

def resize_to_224(x):
    mat = np.reshape(x, (48, 48))
    src = cv2.resize(mat, dsize=(224, 224))
    src = src.astype(np.float16)
    return src


def getData2(balance_ones=True):
    # images are 48x48 = 2304 size vectors
    # N = 35887
    Yte = []
    Xte = []
    Xtr = []
    Ytr = []
    Xval = []
    Yval = []

    first = True
    for line in open('./data/fer2013.csv'):
        if first:
            first = False
        else:
            line = line.strip('\n')
            row = line.split(',')
            mat = resize_to_224([int(p)/255.0 for p in row[1].split()])

            if row[2] == 'Training':
                train = []
                train.append(mat)
                train.append(mat)
                train.append(mat)
                Xtr.append(train)
                Ytr.append(int(row[0]))
            elif row[2] == 'PublicTest':
                test = []
                test.append(mat)
                test.append(mat)
                test.append(mat)
                Xte.append(test)
                Yte.append(int(row[0]))
            else:
                val = []
                val.append(mat)
                val.append(mat)
                val.append(mat)
                Xval.append(val)
                Yval.append(int(row[0]))

    Xte, Yte = np.array(Xte, dtype=np.float16), np.array(Yte)
    Xtr, Ytr = np.array(Xtr, dtype=np.float16), np.array(Ytr)
    Xval, Yval = np.array(Xval, dtype=np.float16), np.array(Yval)

    # Xte, Yte = np.array(Xte) / 255.0, np.array(Yte)
    # Xtr, Ytr = np.array(Xtr) / 255.0, np.array(Ytr)
    # Xval, Yval = np.array(Xval) / 255.0, np.array(Yval)

    return Xtr, Ytr, Xte, Yte, Xval, Yval


# Mask the Data

def Reverse(data):
    data[data == 0] = 2
    data -= 1
    return data

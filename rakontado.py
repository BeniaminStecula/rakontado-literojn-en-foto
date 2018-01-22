import os
import numpy as np
from PIL import Image
from sklearn import svm
import matplotlib.pylab as plt
from sklearn import cross_validation


wymiar = 15


def prepare_feature(im):            # tworzy charakterystyczną tablicę dla litery
    # resize
    resize_im = Image.fromarray(np.uint8(im))
    norm_im = np.array(resize_im.resize((30, 30)))  # rozmiar tabeli dla litery

    # remove border
    norm_im = norm_im[3:-3, 3:-3]
    return norm_im.flatten()


def cut_board(im, axis = 0):        # dzieli planszę na pola
    if axis == 0:
        leng = len(im[0])
    else:
        leng = len(im)
    x = [1, leng/wymiar, 2*leng/wymiar, 3*leng/wymiar, 4*leng/wymiar,
         5*leng / wymiar, 6*leng/wymiar, 7*leng/wymiar, 8*leng/wymiar,9*leng/wymiar,
         10*leng / wymiar,11*leng/wymiar, 12*leng/wymiar, 13*leng/wymiar, 14*leng/wymiar,
         leng-1]
    return x


def load_ocr_data(path):    # dzieli zdjęcia treningowe na kategorie
    # create list of all files ending in.jpg
    imlist = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".jpg")]

    # create labels "01"-nazwa płytki z literą - dwa znaki do inta
    labels = [int(imfile.split("/")[-1][0:2]) for imfile in imlist]

    # create features from the images
    features = []
    for imname in imlist:
        im = np.array(Image.open(imname).convert("L"))
        norm_im = prepare_feature(im)
        features.append(norm_im)
    return np.array(features), labels


# DATA - jako tablica znaków
data,target=load_ocr_data("data/")
features, test_features, labels, test_labels = cross_validation.train_test_split(data, target, test_size=0.3, random_state=0)

# Create SVM classificator
clf = svm.SVC(C=1, kernel='linear')

# Fitting
clf.fit(features, labels)

# Reading test photo
imname = "photos/2.JPG"

im = np.array(Image.open(imname).convert("L"))

ax = plt.subplot(111)
ax.imshow(im, cmap=plt.cm.Greys_r)
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

# krawędzie planszy
x = cut_board(im, axis=0)
y = cut_board(im, axis=1)

# crop cells and classify
crops = []          # pola na płytki
for col in range(wymiar):
    for row in range(wymiar):
        crop = im[int(y[col]):int(y[col+1]), int(x[row]):int(x[row+1])] # pole na płytkę
        crops.append(prepare_feature(crop.copy()))

pred_scra = clf.predict(crops)            # nie kwadrat
scra = pred_scra.reshape(wymiar, wymiar) # kwadrat

print(pred_scra)
print("Result:")
print(scra)

jest = 0
literaA = 0
for i in range(len(scra)):
    for j in range(len(scra[0])):
        if (scra[i][j]>0 and scra[i][j]<31):
            jest+=1
        if (scra[i][j] == 1):
            literaA += 1
print(jest, '/100')
print('ile A: ', literaA, '/9')
plt.show()

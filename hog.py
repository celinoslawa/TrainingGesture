#!/usr/bin/env python
import numpy as np
import cv2
import json
np.set_printoptions(threshold=np.nan)

class StatModel(object):
    def load(self, fn):
        self.model.load(fn)  # Known bug: https://github.com/opencv/opencv/issues/4969
    def save(self, fn):
        self.model.save(fn)

class SVM(StatModel):
    def __init__(self, C = 12.5, gamma = 0.50625):
        self.model = cv2.ml.SVM_create()
        self.model.setGamma(gamma)
        self.model.setC(C)
        self.model.setKernel(cv2.ml.SVM_RBF)
        self.model.setType(cv2.ml.SVM_C_SVC)

    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    def predict(self, samples):

        return self.model.predict(samples)[1].ravel()


#text_file = open("HOG.txt", "w")
#size of image
winSize = (60, 100)
blockSize = (40, 40)
#determines the overlap between neighboring blocks and controls the degree of contrast normalization
blockStride = (20, 20)
#The cellSize is chosen based on the scale of the features important to do the classification.
cellSize = (10, 10)
nbins = 9
derivAperture = 1
winSigma = -1.
histogramNormType = 0
L2HysThreshold = 0.2
gammaCorrection = 1
nlevels = 64
#00-180 degrees with sign
signedGradient = True

#defining hog parameters
hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                        histogramNormType, L2HysThreshold, gammaCorrection, nlevels, signedGradient)

##############################  TRAINING
hog_descriptors = []
N = 125
for i in range(0,10):
    gest = i
    print('GEST : %d' %gest)
    #print(N)

    for j in range(1, N):
        #print(j)
        img = cv2.imread('%d' %gest + 'mask/dys_%d.jpg' %j, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(img, (60, 100), interpolation=cv2.INTER_AREA)
        hog_descriptors.append(hog.compute(mask))
        #text_file.write("%s" %hog.compute(mask))

#print('Write to file ...')
#text_file.write( str(hog_descriptors))

#Remove single-dimensional entries from the shape of an array.
hog_descriptors = np.squeeze(hog_descriptors)
responses = np.int32(np.repeat(np.arange(10),N-1)[:,np.newaxis])
print('Responses: %s' %str(responses))
print(responses.shape)
responsesList = responses.tolist()
with open("responses.json", "w") as f:
    json.dump(obj=responsesList, fp=f)

hog_descriptorsList = hog_descriptors.tolist()
with open("hog_descriptors.json", "w") as f:
    json.dump(obj=hog_descriptorsList, fp=f)

print(hog_descriptors.shape)

print('Training SVM model ...')
model = SVM()
model.train(hog_descriptors, responses)

print('Saving SVM model ...')
model.save('digits_svm.dat')

###################################### TESTING
test_N = 84
hog_descriptors_test = []
for j in range(1, test_N):
    # print(j)
    imgT = cv2.imread('testmask/dys_%d.jpg' % j, cv2.IMREAD_GRAYSCALE)
    maskT = cv2.resize(imgT, (60, 100), interpolation=cv2.INTER_AREA)
    hog_descriptors_test.append(hog.compute(maskT))
hog_descriptors_test = np.squeeze(hog_descriptors_test)
print('Write to file ...')
#text_file.write( str(hog_descriptors_test))

label_test=[ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.,  0.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,
  2.,  2.,  3.,  3.,  3.,  3.,  3.,  4.,  4.,  4.,  4.,  4.,  5.,  5.,  5.,  6.,  6.,  6.,
  6.,  6.,  6.,  6.,  6.,  6.,  6.,  6.,  6.,  7.,  7.,  7.,  7.,  7.,  7.,  7.,  7.,  7.,
  8.,  8.,  8.,  8.,  8.,  8.,  8.,  8.,  8.,  9.,  9.,  9.,  9.,  9.,  9.,  9.,  9.,  9.,
  9.,  9.,  9.,  9.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]

resp = model.predict(hog_descriptors_test)
print(resp.shape)
print('Resp: %s' %str(resp))

err = (label_test != resp).mean()
print('Accuracy: %.2f %%' % ((1 - err) * 100))


#text_file.write(str(hog_descriptors))
#text_file.close()
import scipy.io as spio
from scipy.ndimage import imread
from scipy.misc import imresize
import numpy as np
import matplotlib.pyplot as plt
import pdb
from random import shuffle

def readList(filename):
    f = open(filename, 'r')
    allLines = f.readlines()
    f.close()
    #Remove newlines from all lines
    return [line[:-1] for line in allLines]

"""
An object that handles data input
"""
class dataObj:
    imgIdx = 0
    inputShape = (224, 224, 3)
    maxDim = 0

    #Constructor takes a text file containing a list of absolute filenames
    #Will calculate the mean/std of image for normalization
    #resizeMethod takes 3 different types of input:
    #"crop" will resize the smallest dimension to inputShape,
    #and crop the other dimension in the center
    #"pad" will resize the largest dimension to inputShape, and pad the other dimension
    #"max" will find the max dimension of the list of images, and pad the surrounding area
    #Additionally, if inMaxDim is set with resizeMethod of "max", it will explicitly set
    #the max dimension to inMaxDim
    def __init__(self, imgList, resizeMethod="crop"):
        self.resizeMethod=resizeMethod
        self.imgFiles = readList(imgList)
        self.numImages = len(self.imgFiles)
        self.shuffleIdx = range(self.numImages)
        shuffle(self.shuffleIdx)
        #This function will also set self.maxDim
        self.getMeanVar()
        if(self.resizeMethod=="crop"):
            pass
        elif(self.resizeMethod=="pad"):
            pass
        elif(self.resizeMethod=="max"):
            self.inputShape=(self.maxDim, self.maxDim, 3)
        else:
            print "Method ", resizeMethod, "not supported"
            assert(0)

    #Explicitly sets the mean and standard deviation for normalization
    def setMeanVar(self, inMean, inStd):
        self.mean = inMean
        self.std = inStd

    #Explicitly sets the max dim for the input shape.
    def setMaxDim(self, inMaxDim):
        if(self.maxDim > inMaxDim):
            print "Error, input maxDim (", inMaxDim, ") is smaller than the biggest dimension in input images (", self.maxDim, ")"
            assert(0)
        self.maxDim = inMaxDim
        self.inputShape=(self.maxDim, self.maxDim, 3)

    #Calculates the mean and standard deviation from the images
    #Will also calculate the max dimension of image
    def getMeanVar(self):
        s = 0
        num = 0
        for f in self.imgFiles:
            img = (imread(f).astype(np.float32)/256)
            [ny, nx, nf] = img.shape
            if(ny > self.maxDim):
                self.maxDim = ny
            if(nx > self.maxDim):
                self.maxDim = nx
            s += np.sum(img)
            num += img.size
        self.mean = s / num
        print "img mean: ", self.mean
        ss = 0
        for f in self.imgFiles:
            img = (imread(f).astype(np.float32)/256)
            ss += np.sum(np.power(img-self.mean, 2))
        self.std = np.sqrt(float(ss)/num)
        print "depth std: ", self.std
        print "maxDim: ", self.maxDim

    #Function to resize image to inputShape
    def resizeImage(self, inImage):
        (ny, nx, nf) = inImage.shape
        if(self.resizeMethod == "crop"):
            if(ny > nx):
                #Get percentage of scale
                scale = float(self.inputShape[1])/nx
                targetNy = int(round(ny * scale))
                scaleImage = imresize(inImage, (targetNy, self.inputShape[1]))
                cropTop = (targetNy-self.inputShape[0])/2
                outImage = scaleImage[cropTop:cropTop+self.inputShape[0], :, :]
            elif(ny <= nx):
                #Get percentage of scale
                scale = float(self.inputShape[0])/ny
                targetNx = int(round(nx * scale))
                scaleImage = imresize(inImage, (self.inputShape[0], targetNx))
                cropLeft = (targetNx-self.inputShape[1])/2
                outImage = scaleImage[:, cropLeft:cropLeft+self.inputShape[1], :]
        elif(self.resizeMethod == "pad"):
            if(ny > nx):
                #Get percentage of scale
                scale = float(self.inputShape[0])/ny
                targetNx = int(round(nx * scale))
                scaleImage = imresize(inImage, (self.inputShape[0], targetNx))
                padLeft = (self.inputShape[1]-targetNx)/2
                padRight = self.inputShape[1] - (padLeft + targetNx)
                outImage = np.pad(scaleImage, ((0, 0), (padLeft, padRight), (0, 0)), 'constant')
            elif(ny <= nx):
                #Get percentage of scale
                scale = float(self.inputShape[1])/nx
                targetNy = int(round(ny * scale))
                scaleImage = imresize(inImage, (targetNy, self.inputShape[1]))
                padTop = (self.inputShape[0]-targetNy)/2
                padBot = self.inputShape[0] - (padTop + targetNy)
                outImage = np.pad(scaleImage, ((padTop, padBot), (0, 0), (0, 0)), 'constant')
        elif(self.resizeMethod=="max"):
            #We pad entire image with 0
            assert(ny <= self.inputShape[0])
            assert(nx <= self.inputShape[1])
            padTop   = (self.inputShape[0]-ny)/2
            padBot   = self.inputShape[0]-(padTop+ny)
            padLeft  = (self.inputShape[1]-nx)/2
            padRight = self.inputShape[1]-(padLeft+nx)
            outImage = np.pad(inImage, ((padTop, padBot), (padLeft, padRight), (0, 0)), 'constant')
        else:
            print "Method ", resizeMethod, "not supported"
            assert(0)
        return outImage

    #Reads image provided in the argument, resizes, and normalizes image
    #Also parses ground truth label from path and sets a one-hot 2 dimensional vector
    #Returns the image and ground truth
    def readImage(self, filename):
        image = imread(filename)
        image = ((self.resizeImage(image).astype(np.float32)/256)-self.mean)/self.std
        gt = np.zeros((2))
        s = filename.split('/')[-2]
        if(s == 'Negative'):
            gt[0] = 1
        elif(s == 'Positive'):
            gt[1] = 1
        else:
            print("Unexpected keyword")
            assert(0)
        return(image, gt)

    #Grabs the next image in the list. Will shuffle images when rewinding
    def nextImage(self):
        imgFile = self.imgFiles[self.shuffleIdx[self.imgIdx]]
        #Update imgIdx
        self.imgIdx = (self.imgIdx + 1) % len(self.imgFiles)
        if(self.imgIdx >= self.numImages):
            print "Rewinding"
            self.imgIdx = 0
            shuffle(range(self.numImages))
        return self.readImage(imgFile)

    #Get all segments of current image. This is what evaluation calls for testing
    def allImages(self):
        outData = np.zeros((self.numImages, self.inputShape[0], self.inputShape[1], self.inputShape[2]))
        outGt = np.zeros((self.numImages, 2))
        for i, imgFile in enumerate(self.imgFiles):
            data = self.readImage(imgFile)
            outData[i, :, :, :] = data[0]
            outGt[i, :] = data[1]
        return (outData, outGt)

    #Gets numExample images and stores it into an outer dimension.
    #This is what TF object calls to get images for training
    def getData(self, numExample):
        outData = np.zeros((numExample, self.inputShape[0], self.inputShape[1], 3))
        outGt = np.zeros((numExample, 2))
        for i in range(numExample):
            data = self.nextImage()
            outData[i, :, :, :] = data[0]
            outGt[i, :] = data[1]
        return (outData, outGt)


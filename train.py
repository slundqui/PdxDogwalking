import matplotlib
matplotlib.use('Agg')
from dataObj.dogwalker import dataObj
from tf.imageClassification import imageClassification
from plot.roc import makeRocCurve
import numpy as np
import pdb
import os

#Input vgg file for preloaded weights
vggFile = "/home/sheng/mountData/pretrain/imagenet-vgg-f.mat"

#Paths to list of filenames
trainImageList = "/home/sheng/mountData/datasets/CropsForObjectClassifier/Dog/train.txt"
testImageList = "/home/sheng/mountData/datasets/CropsForObjectClassifier/Dog/test.txt"

#Base output directory
outDir = "/home/sheng/mountData/dogwalk/"
#Inner run directory
runDir = outDir + "/dogPad_run0/"
#output plots directory
plotDir = runDir + "plots/"

if not os.path.exists(runDir):
   os.makedirs(runDir)

if not os.path.exists(plotDir):
   os.makedirs(plotDir)

#Flag for loading weights from checkpoint
load = False
loadFile = outDir + "/saved/dog-saved.ckpt"

#Get object from which tensorflow will pull data from
trainDataObj = dataObj(trainImageList, resizeMethod="pad")
testDataObj = dataObj(testImageList, resizeMethod="pad")

testDataObj.setMeanVar(trainDataObj.mean, trainDataObj.std)
##If resizeMethod == "max", run this code to match maxDims
#if(trainDataObj.maxDim >= testDataObj.maxDim):
#    testDataObj.setMaxDim(trainDataObj.maxDim)
#else:
#    trainDataObj.setMaxDim(testDataObj.maxDim)

#Allocate tensorflow object
tfObj = imageClassification(trainDataObj.inputShape, vggFile)

#Load checkpoint if flag set
if(load):
   tfObj.loadModel(loadFile)
else:
   tfObj.initSess()

#For tensorboard
tfObj.writeSummary(runDir + "/tfout")

print "Done init"

saveFile = runDir + "/dog-model"
#Training
for i in range(1000):
   #Evaluate test frame, providing gt so that it writes to summary
   (evalData, gtData) = testDataObj.allImages()
   tfObj.evalModel(evalData, gtData)
   print "Done test eval"
   #Train
   tfObj.trainModel(trainDataObj, 10, saveFile, pre=True)

print "Done run"

tfObj.closeSess()







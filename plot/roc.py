import numpy as np
import matplotlib.pyplot as plt

#Function to compute the output class vector given a threshold
def calcClass(inScores, threshold):
   #astype casts boolean to 0 and 1
   return (inScores >= threshold).astype(int)

#Function to calculate accuracy given a array of ests and array of gts
#estClass and gtClass is a [M] array, where M is number of instances
#Returns the accuracy, precision, recall, and false positive rate
def calcStats(estClass, gtClass):
   numInstance = len(estClass)
   assert(len(gtClass == numInstance))

   #Get TP, FP, TN, FN numbers
   TP = np.sum((gtClass[np.nonzero(estClass == 1)] == 1).astype(int))
   TN = np.sum((gtClass[np.nonzero(estClass == 0)] == 0).astype(int))
   FP = np.sum((gtClass[np.nonzero(estClass == 1)] == 0).astype(int))
   FN = np.sum((gtClass[np.nonzero(estClass == 0)] == 1).astype(int))

   #We need 4 metrics: Accuracy, Precision, Recall, and FPR
   accuracy = float(TP + TN) / float(numInstance)
   precision = float(TP) / float(TP + FP)
   recall = float(TP) / float(TP + FN)
   fpr = float(FP)/float(FP + TN)

   return(accuracy, precision, recall, fpr)

#Function to make a roc curve given output scores and ground truth
def makeRocCurve(scores, gt, numThreshBins=1000, plotsOutDir="./"):
   #Get max and min of scores
   minScore = np.min(scores)
   maxScore = np.max(scores)
   scoreStep = (maxScore - minScore) / numThreshBins
   #Build array of testable threshold values
   threshVals = np.arange(minScore, maxScore, scoreStep)
   TPR = np.zeros((numThreshBins))
   FPR = np.zeros((numThreshBins))

   for (i, thresh) in enumerate(threshVals):
      #Calculate class based on threshold
      estClass = calcClass(scores, thresh)
      #Calculate stats and append to vectors
      (a, p, r, f) = calcStats(estClass, gt)
      TPR[i] = r
      FPR[i] = f

   #Calculate area under curve using trapz
   auc = np.trapz(TPR[::-1], FPR[::-1])

   plt.figure()
   plt.plot(FPR, TPR, label="roc")
   plt.plot([0, 1], [0, 1], 'r--', label="chance")
   plt.title("ROC curve (AUC = " + str(auc) + ")")
   plt.xlabel("False Positive Rate")
   plt.ylabel("True Positive Rate")
   #plt.show()
   plt.savefig(plotsOutDir + "roc.jpg")

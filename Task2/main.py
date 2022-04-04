from operator import indexOf
import numpy as np
import matplotlib.pyplot as mplot
import knn_prediction_regression

###
#   Program created by Robin Svensson, rs223dj - rs223dj@student.lnu.se
#
#   MSE calculations ended with the results:
#   Training:
#     k1: 0,     k3: 18.04, k5: 22.52, k7: 23.56, k9: 24.41, k11: 26.41
#   Test:
#     k1: 49.23, k3: 31.58, k5: 28.48, k7: 29.23, k9: 27.70, k11: 30.35
#
#   I would have gone with k9, as this presents the lowest amount of errors for both cases.
#   This could be seen in these results but was also plotted to see where they got the lowest cumulative result.
#   Was also calculated to be lowest amount of errors. 
###

def main():
    full_Array = np.loadtxt("./Task2/polynomial200.csv", dtype=np.float64, delimiter=",")
    full_Array = np.split(full_Array, 2)
    training_set = full_Array[0]
    test_set = full_Array[1]
    
    predictor = knn_prediction_regression.knn_prediction()
    xx = np.linspace(1,25, num=100)
    
    # Calculations regarding the prediction of the regline.
    yvals = []
    mses = []
    kvals = [1,3,5,7,9,11]
    for k in kvals:
      yy = predictor.predict(training_set, training_set[:,0], k)
      regline = predictor.predict(training_set, xx, k)
      yvals.append(regline)
      mses.append(predictor.mse(training_set[:,1], yy))
    
    # Calculations regarding test results vs the training set.
    test_mses = []
    test_pred = knn_prediction_regression.knn_prediction()
    print("Errors using MSE error for test:")
    for k in kvals:
      tt = test_pred.predict(training_set, test_set[:,0], k)
      mse = test_pred.mse(test_set[:,1] ,tt)
      test_mses.append(mse)
      print("k =", k, "mse:", mse)
    
    # Calculate the effectiveness of predictions.     
    # for i in range(6):
    #   print("k", kvals[i], ": ", test_mses[i] - mses[i])
    # plotMSES(kvals, mses, test_mses)
    plotAll(xx, yvals, training_set)
    
def plotMSES(kvals, mses, test_mses):
    mplot.plot(kvals, mses)
    mplot.plot(kvals, test_mses)
    mplot.show()
    
def plotAll(xvals, yvals, predictions):
  fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = mplot.subplots(2,3)
  ax1.plot(xvals, yvals[0])
  ax1.scatter(predictions[:,0], predictions[:,1],c="g")
  ax1.set_title("k = 1, MSE = 0")
  
  ax2.plot(xvals, yvals[1])
  ax2.scatter(predictions[:,0], predictions[:,1],c="g")
  ax2.set_title("k = 3, MSE = 18.04")
  
  ax3.plot(xvals, yvals[2])
  ax3.scatter(predictions[:,0], predictions[:,1],c="g")
  ax3.set_title("k = 5, MSE = 22.52")
  
  ax4.plot(xvals, yvals[3])
  ax4.scatter(predictions[:,0], predictions[:,1],c="g")
  ax4.set_title("k = 7, MSE = 23.56")
  
  ax5.plot(xvals, yvals[4])
  ax5.scatter(predictions[:,0], predictions[:,1],c="g")
  ax5.set_title("k = 9, MSE = 24.41")
  
  ax6.plot(xvals, yvals[5])
  ax6.scatter(predictions[:,0], predictions[:,1],c="g")
  ax6.set_title("k = 11, MSE = 26.41")
  mplot.show()
    
if __name__ == "__main__":
  main()
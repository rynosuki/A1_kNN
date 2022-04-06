import numpy as np
import matplotlib.pyplot as mplot
import T2_prediction_regression as knn_prediction_regression

###
#   Program created by Robin Svensson, rs223dj - rs223dj@student.lnu.se
#       
#     Question 4.
#   MSE calculations ended with the results:
#   Training:
#     k1: 0,     k3: 18.04, k5: 22.52, k7: 23.56, k9: 24.41, k11: 26.41
#   Test:
#     k1: 49.23, k3: 31.58, k5: 28.48, k7: 29.23, k9: 27.70, k11: 30.35
#
#     Question 5.
#   I would have gone with k9, as this presents the lowest amount of errors for both cases.
#   This could be seen in these results but was also plotted to see where they got the lowest cumulative result.
#   Was also calculated to be lowest amount of errors. 
###

def main():
    full_Array = np.loadtxt("./polynomial200.csv", dtype=np.float64, delimiter=",")
    full_Array = np.split(full_Array, 2)
    training_set = full_Array[0]
    test_set = full_Array[1]
    
    plot1x2(training_set, test_set)
    
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
      print(tt)
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
  fig, ((ax0,ax1,ax2),(ax3,ax4,ax5)) = mplot.subplots(2,3,figsize=(10,10))
  textval = ["k = 1, MSE = 0", "k = 3, MSE = 18.04", "k = 5, MSE = 22.52",
             "k = 7, MSE = 23.56", "k = 9, MSE = 24.41", "k = 11, MSE = 26.41"]
  for i, p in enumerate(fig.axes):
    p.plot(xvals, yvals[i])
    p.scatter(predictions[:,0], predictions[:,1],c="g")
    p.set_title(textval[i])
  mplot.show()
    
def plot1x2(training_set, test_set):
  fig, (ax0,ax1) = mplot.subplots(1,2)
  
  ax0.plot(training_set[:,0], training_set[:,1], "ro")
  ax0.set_title("Training Set")
  
  ax1.plot(test_set[:,0],test_set[:,1], "bo")
  ax1.set_title("Test Set")
  
if __name__ == "__main__":
  main()
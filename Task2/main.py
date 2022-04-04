from operator import indexOf
import numpy as np
import matplotlib.pyplot as mplot
import knn_prediction_regression

def main():
    full_Array = np.loadtxt("./Task2/polynomial200.csv", dtype=np.float64, delimiter=",")
    full_Array = np.split(full_Array, 2)
    training_set = full_Array[0]
    test_set = full_Array[1]
    
    predictor = knn_prediction_regression.knn_prediction(training_set)
    xx = np.linspace(1,25, num=200)
    yvals = []
    for k in [1,3,5,7,9,11]:
      yy = predict(training_set, training_set[:,0], k)
      yy = np.split(yy,2)[0]
      regline = predict(training_set, xx, k)
      yvals.append(regline)
    
    test_pred = knn_prediction_regression.knn_prediction(test_set)
    print("Errors using MSE error for test:")
    for k in [1,3,5,7,9,11]:
      tt = predict(training_set, test_set[:,0], k)
      tt = np.split(tt,2)[0]
      print(test_pred.mse(tt))
    plotAll(xx, yvals, training_set)
    plotAll(xx, yvals, test_set)

def predict(data, xaxis, k):
  data = data[data[:, 0].argsort()]
  yy = np.linspace(0,0,num = 200)
  x_array = np.array(data[:,0])
  index = 0
  for i in xaxis:
    h = abs(x_array - i)
    for d in h:
      if d == 0:
        h = np.delete(h, indexOf(h, d))
    shortest = np.argpartition(h,k)
    for j in range(k):
      yy[index] += data[shortest[j]][1]
    yy[index] = yy[index] / k
    index+=1
  return yy
    
def plotAll(xvals, yvals, predictions):
  fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = mplot.subplots(2,3)
  ax1.plot(xvals, yvals[0])
  ax1.scatter(predictions[:,0], predictions[:,1],c="g")
  ax1.set_title("k = 1, MSE = 23.42")
  
  ax2.plot(xvals, yvals[1])
  ax2.scatter(predictions[:,0], predictions[:,1],c="g")
  ax2.set_title("k = 3, MSE = 17.98")
  
  ax3.plot(xvals, yvals[2])
  ax3.scatter(predictions[:,0], predictions[:,1],c="g")
  ax3.set_title("k = 5, MSE = 22.76")
  
  ax4.plot(xvals, yvals[3])
  ax4.scatter(predictions[:,0], predictions[:,1],c="g")
  ax4.set_title("k = 7, MSE = 23.88")
  
  ax5.plot(xvals, yvals[4])
  ax5.scatter(predictions[:,0], predictions[:,1],c="g")
  ax5.set_title("k = 9, MSE = 24.35")
  
  ax6.plot(xvals, yvals[5])
  ax6.scatter(predictions[:,0], predictions[:,1],c="g")
  ax6.set_title("k = 11, MSE = 28.33")
  mplot.show()
    
if __name__ == "__main__":
  main()
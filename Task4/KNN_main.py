from time import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets
from sklearn.metrics import accuracy_score, confusion_matrix

def main():
  data = np.genfromtxt("Task4\microchips.csv", delimiter=",")
  X = data[:, :2]
  y = data[:, 2]
  yx = np.array([np.array([-0.3, 1.0]), np.array([-0.5, -0.1]), np.array([0.6, 0.0])])
  
  y_preds = []
  err_preds = []
  decision_boundaries = []
  t = time()
  for k in [1,3,5,7]:
    clf = neighbors.KNeighborsClassifier(k, metric = "euclidean", p = 2)
    clf.fit(X,y)
    y_pred = clf.predict(yx)
    print("k:",k,"values:",y_pred)
    y_preds.append(y_pred)
    
    err_pred = clf.predict(X)
    err_preds.append(accuracy_score(y, 1-err_pred))
    print("Error rate for k:",k,1-accuracy_score(y, err_pred))
    
    decision_boundaries.append(createDecisionBoundary(data, clf))
  print(time()-t)  
  plot_all(data, y_preds, err_preds, [1,3,5,7], decision_boundaries)
  
def plot_all(data, y_preds, err_preds, kvals, decision_boundaries):
  plt.scatter(data[:,0], data[:,1], c=data[:,2])
  plt.show()
  
  fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 5))

  ax1.imshow(decision_boundaries[0], origin='lower', extent=(min(data[:,0]), max(data[:,0]), min(data[:,1]), max(data[:,1])))
  ax1.scatter(data[:,0], data[:,1], c=data[:,2], edgecolors='r')
  ax1.set_title(("k =", kvals[0], "error rate =", err_preds[0]))
  
  ax2.imshow(decision_boundaries[1], origin='lower', extent=(min(data[:,0]), max(data[:,0]), min(data[:,1]), max(data[:,1])))
  ax2.scatter(data[:,0], data[:,1], c=data[:,2], edgecolors='r')
  ax2.set_title(("k =", kvals[1], "error rate =", err_preds[1]))
  
  ax3.imshow(decision_boundaries[2], origin='lower', extent=(min(data[:,0]), max(data[:,0]), min(data[:,1]), max(data[:,1])))
  ax3.scatter(data[:,0], data[:,1], c=data[:,2], edgecolors='r')
  ax3.set_title(("k =", kvals[2], "error rate =", err_preds[2]))
  
  ax4.imshow(decision_boundaries[3], origin='lower', extent=(min(data[:,0]), max(data[:,0]), min(data[:,1]), max(data[:,1])))
  ax4.scatter(data[:,0], data[:,1], c=data[:,2], edgecolors='r')
  ax4.set_title(("k =", kvals[3], "error rate =", err_preds[3]))
  plt.show()
  
def createDecisionBoundary(array, predictor, gridsize = 1000):
    min_x, max_x = min(array[:,0]), max(array[:,0])
    min_y, max_y = min(array[:,1]), max(array[:,1])
    x_axis = np.linspace(min_x, max_x, gridsize)
    y_axis = np.linspace(min_y, max_y, gridsize)
    xx, yy = np.meshgrid(x_axis, y_axis)
    cells = np.stack([xx.ravel(), yy.ravel()], axis=1)
    grid = predictor.predict(cells).reshape(gridsize, gridsize)
    return grid    
  
if __name__ == "__main__":
  main()
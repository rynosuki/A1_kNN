from matplotlib import pyplot as plt
import numpy as np

class knn_prediction:
  def __init__(self, X, y):
    self.X = X
    self.y = y
    
  def eucledian(self,point, X):
    return np.linalg.norm(np.array(point) - X[:,:], axis=1)
  
  def nearest_neighbors(self, distance_val, k):
    return np.argpartition(distance_val, k)
    
  def predict(self, pred, k):
    val = []
    for i in pred:
      distances = self.eucledian(i, self.X)
      nearest_neighbor = self.nearest_neighbors(distances, k)
      result = []
      for j in range(k):
        result.append(self.y[nearest_neighbor[j]])
      g = np.bincount(result)
      val.append(np.argmax(g))
    return np.array(val, dtype=np.uint32)
  
  def accuracy(prediction, expected):
    t = np.equal(prediction, expected)
    acc = [0,0]
    for i in t:
      if i:
        acc[0] += 1
      else:
        acc[1] += 1
    return (acc[0]/len(prediction))*100
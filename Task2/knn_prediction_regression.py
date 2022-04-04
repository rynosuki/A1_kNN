from operator import indexOf
import numpy as np

class knn_prediction:
  def mse(self, values, point):
    return np.square(np.subtract(values,point)).mean()
   
  def eucledian(self, values, point):
    return np.linalg.norm(values - np.array(point))
  
  def manhattan(self, values, point):
    return np.sum(np.abs(values - np.array(point)), axis=1)
  
  def nearest_neighbors(self, distance_val, k):
    return np.argpartition(distance_val, k)
    
  def predict(self, values, pred, k):
    y_values = []
    for p in range(len(values)):
      y_values.append(0)
      distances = np.abs(values[:,0] - pred[p])
      nearest_neighbours = self.nearest_neighbors(distances, k)
      for n in range(k):
        y_values[p] += values[nearest_neighbours[n]][1]
    y_values = np.array(y_values) / k
    return y_values
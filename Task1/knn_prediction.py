import numpy as np

class knn_prediction:
  def __init__(self, values):
    self.values = values
   
  def eucledian(self, values, point):
    return np.linalg.norm(values - np.array(point), axis = 1)
  
  def nearest_neighbors(self, distance_val, k):
    return np.argpartition(distance_val, k)
    
  def predict(self, pred, k):
    values = []
    for i in pred:
      distances = self.eucledian(self.values[:,:2], i)
      nearest_neighbor = self.nearest_neighbors(distances, k)
      result = 0
      for j in range(k):
        result += self.values[:,2][nearest_neighbor][j]
      if result / k > 0.5:
        values.append(1)
      else:
        values.append(0)
    return np.array(values)
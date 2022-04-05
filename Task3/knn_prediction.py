import numpy as np

class knn_prediction:
  def __init__(self, values, results):
    self.values = values
    self.results = results
  
  def eucledian(self, point):
    return np.linalg.norm(np.array(point) - self.values[:,:], axis=1)
  
  def nearest_neighbors(self, distance_val, k):
    return np.argpartition(distance_val, k)
    
  def predict(self, pred, k):
    val = []
    for i in pred:
      distances = self.eucledian(i)
      nearest_neighbor = self.nearest_neighbors(distances, k)
      result = []
      for j in range(k): # k
        result.append(self.results[nearest_neighbor[j]])
      g = np.bincount(result)
      val.append(np.argmax(g))
    return np.array(val, dtype=np.uint32)
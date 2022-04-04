import numpy as np

class error_calculator:
  def eucledian(self, values, point):
    return np.linalg.norm(values - np.array(point), axis = 1)
  
  def nearest_neighbors(self, distance_val, k):
    return np.argpartition(distance_val, k)
  
  def calculate_errors(self):
    for k in self.k_values:
      self.errors[k] = 0
      index = 0
      for i in self.values:
        distances = self.eucledian(self.values[:,:2], i[:2])
        nearest_neighbor = self.nearest_neighbors(distances, k)
        result = 0
        for j in range(k):
          result += self.values[:,2][nearest_neighbor[j]]
        if result / k >= 0.5 and i[2] != 1 or result / k < 0.5 and i[2] != 0:
          self.errors[k] += 1
        index += 1
      
  
  def __init__(self, values, k_values):
    self.values = values
    self.value_length = len(values)
    self.k_values = k_values
    self.errors = {}
    
  def get_errors(self):
    return self.errors
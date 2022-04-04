import numpy as np

class knn_prediction:
  def __init__(self, values):
    self.values = values
    
  def mse(self, point):
    return np.square(np.subtract(self.values[:,1], point)).mean()
   
  def eucledian(self, values, point):
    return np.linalg.norm(values - np.array(point), axis = 1)
  
  def manhattan(self, values, point):
    return np.sum(np.abs(values - np.array(point)), axis=1)
  
  def nearest_neighbors(self, distance_val, k):
    return np.argpartition(distance_val, k)
    
  def predict(self, pred, k):
    values = np.array([])
    ind = 0
    for i in pred[:,0]:
      values = np.append(values, ind)
      h = abs(self.values[:,0] - i)
      index = 0
      for l in h:
        if l == 0:
          h = np.delete(h, index)
        index += 1
      shortest = np.argpartition(h,k)
      for n in range(k):
        values[ind] += pred[shortest[n]][1]
      ind += 1
    values = np.divide(values, k)
    return values
  
  def nearest_by_X(self, x_value, k_value, plus_value=0.0001):
  
    x = self.values[:,0]

    marg_value= 0.0
    array = np.where( np.logical_and ( x < x_value + marg_value, x > x_value - marg_value ) )[0]
    while array.shape[0] < k_value:
      marg_value += plus_value
      array = np.where( np.logical_and (x < x_value + marg_value, x > x_value - marg_value ) )[0]
    return array
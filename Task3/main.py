from time import time
from keras.datasets import mnist

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from knn_prediction import knn_prediction

def main():
  import numpy as np
  from PIL import Image
  
  (train_X, train_y), (test_X, test_y) = mnist.load_data()
  train_shaped = train_X.reshape(60000,784)
  # train_shaped = np.array_split(train_shaped,60)[0]
  # train_y_shaped = np.array_split(train_y,60)[0]
  test_shaped = test_X.reshape(10000,784)
  test_shaped = np.array_split(test_shaped,100)[0]
  test_y_shaped = np.array_split(test_y,100)[0]
  
  test_shaped_2 = test_X.reshape(10000,784)
  test_shaped_2 = np.array_split(test_shaped_2,50)[0]
  test_y_shaped_2 = np.array_split(test_y,50)[0]
  
  predictor = knn_prediction(train_shaped, train_y)
  cur = time()
  pred_y = predictor.predict(test_shaped,5)
  print(time()-cur)
  cur = time()
  pred_y = predictor.predict(test_shaped_2,5)
  print(time()-cur)
  
  # acc_train = []
  # acc_test = []
  # kvals = [1,3,5,7,9,11]

  # for k in kvals:
  #   pred_y = predictor.predict(test_shaped,k)
    
  # t = np.equal(pred_y, test_y_shaped)
  # acc = [0,0]
  # for i in t:
  #   if i:
  #     acc[0] += 1
  #   else:
  #     acc[1] += 1
  # print(acc[0]/len(pred_y))
    
  # for k in kvals:
  #   pred_y = predictor.predict(test_shaped, k)
  #   print(accuracy_score(test_y, pred_y))

  # xx = np.linspace(1,100,100)
  # plt.plot(kvals, acc_train)
  # plt.plot(kvals, acc_test)
  # plt.legend(["Acc - Train", "Acc - Test"])
  # plt.show()

  # for i in range(9):  
  #   plt.subplot(330 + 1 + i)
  #   plt.imshow(train_X[i], cmap=plt.get_cmap('gray'))
  # plt.show()

if __name__ == "__main__":
  main()
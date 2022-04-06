from keras.datasets import mnist
import numpy as np
from T3_knn_prediction import knn_prediction

def main():
  
# # # # # # # # # # # # # # # # # # # # # # # #
#     Accuracy of training: 1000 At k: 1      #
#       Accuracy of test: 83 At k: 1          #
#                                             #
#     Accuracy of training: 851 At k: 3       #
#       Accuracy of test: 83 At k: 3          #
#                                             #
#     Accuracy of training: 812 At k: 5       #
#       Accuracy of test: 84 At k: 5          #
#                                             #
#     Accuracy of training: 799 At k: 7       #
#       Accuracy of test: 81 At k: 7          #
#                                             #
#     Accuracy of training: 795 At k: 9       #
#       Accuracy of test: 82 At k: 9          #
#                                             #
#     Accuracy of training: 784 At k: 11      #
#       Accuracy of test: 80 At k: 11         #
# # # # # # # # # # # # # # # # # # # # # # # #
#                                             #
#           K chosen to go for: 5             #
#          Accuracy of test: 96.88%           #
#               For 60000:10000               #
#               For full test                 #
#                                             #
# # # # # # # # # # # # # # # # # # # # # # # #
#          Information gathered from:         #
# https://www.youtube.com/watch?v=TKfTKA5EmmY #
# https://www.youtube.com/watch?v=ZD_tfNpKzHY #
# # # # # # # # # # # # # # # # # # # # # # # #
#Optimizations has been made by removing white#
# pixels if such that every array was white   #  
#   Algorithm runs at a complexity of O(n)    #
# # # # # # # # # # # # # # # # # # # # # # # #
#      Uncomment row 48, 49 and 51,52         # 
#             to do 10000x1000                #
# # # # # # # # # # # # # # # # # # # # # # # #
    
    
    
  (train_X, train_y), (test_X, test_y) = mnist.load_data()
  train_shaped = train_X.reshape(60000,784)/255
  # train_shaped = np.array_split(train_shaped,6)[0]
  # train_y_shaped = np.array_split(train_y,6)[0]
  test_shaped = test_X.reshape(10000,784)/255
  # test_shaped = np.array_split(test_shaped,10)[0]
  # test_y_shaped = np.array_split(test_y,10)[0]
  
  
  print(train_shaped.shape)
  train_shaped = train_shaped.T
  test_shaped = test_shaped.T
  changed_trained = train_shaped
  changed_test = test_shaped
  ls = []
  for i in range(784):
    if sum(train_shaped[i]) + sum(test_shaped[i]) == 0:
      ls.append(i)
  for i in reversed(list(enumerate(ls))):
    changed_trained = np.delete(changed_trained, i[1], 0)
    changed_test = np.delete(changed_test, i[1], 0)
  test_shaped = changed_test.T
  train_shaped = changed_trained.T
  print(train_shaped.shape)
  quit()
  predictor = knn_prediction(train_shaped, train_y)
  kvals = [5]
  for k in kvals:
    pred_y = predictor.predict(test_shaped, k)
    accu_test = predictor.accuracy(pred_y, test_y)
    print(accu_test)
    # pred_y = predictor.predict(train_shaped, k)
    # accu_train = accuracy(pred_y, train_y)
    # acc_train.append(accu_train)
    # print(accu_train)
  
  # cur = time()
  # pred_y = predictor.predict(test_shaped, 5)
  # print(time()-cur)
  # print("Accuracy of test:", accuracy(pred_y, test_y_shaped), "At k:", 5)

if __name__ == "__main__":
  main()
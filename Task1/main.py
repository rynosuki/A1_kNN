from time import time
import numpy as np
import matplotlib.pyplot as mplot

from error_calculator import error_calculator
from knn_prediction import knn_prediction

def main():
    array = np.loadtxt("./Task1/microchips.csv", dtype=np.float64, delimiter=",")
    predictor = knn_prediction(array)

    testchips = [[-0.3, 1.0],[-0.5,-0.1],[0.6,0.0]]

    kvals = [1,3,5,7]
    result = []
    for p in testchips:
        for k in kvals:
            prediction = predictor.predict([p],k)
            result.append(prediction.sum())
    printResults(kvals, testchips, result)
    
    errorcalc = error_calculator(array, kvals)
    errorcalc.calculate_errors()
    errors = errorcalc.get_errors()
    print("\n Errors are:")
    for err in errors:
        print(err, "=", (errors[err] / len(array)))
    
    cur = time()
    decision_boundary = []
    
    for k in kvals:
        decision_boundary_k = createDecisionBoundary(array, k, predictor)
        decision_boundary.append(decision_boundary_k)
    print(time() - cur)
    plotAll(decision_boundary, array)

def printResults(kvals, testchips, result):
    for i in range(len(kvals)):
        print("k = " + str(kvals[i]))
        for k in range(len(testchips)):
            if result[i + 4*k] == 1:
                print("chip" + str((k+1)) + " " + str(testchips[k]) + " ==> PASS")
            else:
                print("chip" + str((k+1)) + " " + str(testchips[k]) + " ==> FAIL")
   
def createDecisionBoundary(array, k, predictor):
    min_x, max_x = min(array[:,0]), max(array[:,0])
    min_y, max_y = min(array[:,1]), max(array[:,1])
    gridsize = 100
    x_axis = np.linspace(min_x, max_x, gridsize)
    y_axis = np.linspace(min_y, max_y, gridsize)
    xx, yy = np.meshgrid(x_axis, y_axis)
    cells = np.stack([xx.ravel(), yy.ravel()], axis=1)
    grid = predictor.predict(cells, k).reshape(gridsize, gridsize)
    return grid    

def plotAll(decision_boundary, array):
    fig, ((ax1, ax2), (ax3, ax4)) = mplot.subplots(2, 2, figsize=(10, 5))

    ax1.imshow(decision_boundary[0], origin='lower', extent=(min(array[:,0]), max(array[:,0]), min(array[:,1]), max(array[:,1])))
    ax1.scatter(array[:,0], array[:,1], c=array[:,2], edgecolors='r')
    ax1.set_title("k = 1, error rate = 0.381")
    
    ax2.imshow(decision_boundary[1], origin='lower', extent=(min(array[:,0]), max(array[:,0]), min(array[:,1]), max(array[:,1])))
    ax2.scatter(array[:,0], array[:,1], c=array[:,2], edgecolors='r')
    ax2.set_title("k = 3, error rate = 0.28")
    
    ax3.imshow(decision_boundary[2], origin='lower', extent=(min(array[:,0]), max(array[:,0]), min(array[:,1]), max(array[:,1])))
    ax3.scatter(array[:,0], array[:,1], c=array[:,2], edgecolors='r')
    ax3.set_title("k = 5, error rate = 0.263")
    
    ax4.imshow(decision_boundary[3], origin='lower', extent=(min(array[:,0]), max(array[:,0]), min(array[:,1]), max(array[:,1])))
    ax4.scatter(array[:,0], array[:,1], c=array[:,2], edgecolors='r')
    ax4.set_title("k = 7, error rate = 0.263")
    mplot.show()

if __name__ == "__main__":
    main()
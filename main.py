import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as mplot
import csv

value = []

with open('microchips.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    initString = reader.__next__()
    print(initString)
    array = np.array([float(initString[0]),float(initString[1])])
    value.append(initString[2])
    print(array)
    for row in reader:
        array = np.vstack((array,np.array([float(row[0]),float(row[1])])))
        value.append(int(row[2]))
testchips = [[-0.3, 1.0],[-0.5,-0.1],[0.6,0.0]]

kvals = [1,3,5,7]
result = []

for p in testchips:
    distances = np.linalg.norm(array - np.array(p), axis=1)
    for k in kvals:
        index_min = np.argpartition(distances, k-1)
        tempres = 0
        for l in range(k):
            tempres += value[index_min[l]]
        if tempres / k > 0.5:
            result.append(1)
        else:
            result.append(0)
print(result)

fig, ((ax1,ax2),(ax3,ax4)) = mplot.subplots(2,2)
ax1.set_title("k = 1")
ax2.set_title("k = 3")
ax3.set_title("k = 5")
ax4.set_title("k = 7")
for i in range(len(value)):
    if(value[i]) == 1:
        ax1.plot(array[i][0], array[i][1], "go")
        ax2.plot(array[i][0], array[i][1], "go")
        ax3.plot(array[i][0], array[i][1], "go")
        ax4.plot(array[i][0], array[i][1], "go")
    else:
        ax1.plot(array[i][0], array[i][1], "ro")
        ax2.plot(array[i][0], array[i][1], "ro")
        ax3.plot(array[i][0], array[i][1], "ro")
        ax4.plot(array[i][0], array[i][1], "ro")

for k in range(len(testchips)):
    if result[4*k] == 1:
        ax1.plot(testchips[k][0], testchips[k][1], "gx")
    else: 
        ax1.plot(testchips[k][0], testchips[k][1], "rx")
    if result[4*k+1] == 1:
        ax2.plot(testchips[k][0], testchips[k][1], "gx")
    else: 
        ax2.plot(testchips[k][0], testchips[k][1], "rx")    
    if result[4*k+2] == 1:
        ax3.plot(testchips[k][0], testchips[k][1], "gx")
    else: 
        ax3.plot(testchips[k][0], testchips[k][1], "rx")
    if result[4*k+3] == 1:
        ax4.plot(testchips[k][0], testchips[k][1], "gx")
    else: 
        ax4.plot(testchips[k][0], testchips[k][1], "rx")
        
mplot.show()
        


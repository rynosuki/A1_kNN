import numpy as np
import matplotlib.pyplot as mplot

def main():
    full_Array = np.loadtxt("./Task2/polynomial200.csv", dtype=np.float64, delimiter=",")
    np.split(full_Array, 100)
    print(full_Array)
if __name__ == "__main__":
  main()
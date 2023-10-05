import numpy as np
from sklearn.neighbors import KDTree, BallTree
import time
from sklearn.datasets import make_blobs

x = np.loadtxt('test-100')
x = x.reshape(100, 5)
tree = BallTree(x, leaf_size=40)
start = time.time()
# a, b = tree.query(x, k=6, return_distance=True, dualtree=False, breadth_first=False)
idx, dist = tree.query(x, k=6)
end = time.time()
print(end - start)

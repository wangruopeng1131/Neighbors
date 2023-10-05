# Neighbors
The C++ version of KDtree and BallTree.

## Description
This code is translated from Scikit-learn. Eigen is used for restoring data.
SSE is supported for computing Euclid Distance.

## Usage
```c++
#include "KDTree.h"
#include "BallTree.h"

KDTree KdTree = KDTree(data, 40);
BallTree ballTree = BallTree(data, 40);
auto [dualBreadth, idx1] = KdTree.query(data, 6, true, true);
auto [ballDualDepth, idx6] = ballTree.query(data, 6, true, false);
```
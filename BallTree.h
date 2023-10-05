//
// Created by ruopeng on 2023/2/14.
//

#ifndef HDBSCAN_BALLTREE_H
#define HDBSCAN_BALLTREE_H

#include <cmath>
#include "Eigen/Dense"
#include "BinaryTree.h"

using namespace std;

class BallTree : public BinaryTree<BallTree>
{
public:
    BallTree(MatrixXd &d, int leaf_size) : BinaryTree<BallTree>(d, leaf_size)
    {
        buildBinaryTree(node_data, 0, 0, n_samples);
    };

    ~BallTree() = default;

    void initNode(deque<NodeData> &node_data_temp, int i_node, int idx_start, int idx_end) noexcept(false)
    {
        double n_points = idx_end - idx_start;
        int i, j;
        double radius, tmp;
        double *this_pt;
        double *centroid = lower_node_bounds_ptr + i_node * n_features;

        // determine Node centroid
        for (int k = 0; k < n_features; ++k)
        {
            centroid[k] = 0.0;
        }

        for (i = idx_start; i < idx_end; ++i)
        {
            this_pt = data_ptr + idx_array_ptr[i] * n_features;
            for (j = 0; j < n_features; ++j)
            {
                centroid[j] += this_pt[j];
            }
        }

        for (j = 0; j < n_features; ++j)
        {
            centroid[j] /= n_points;
        }

        // determine Node radius
        radius = 0;
        for (i = idx_start; i < idx_end; ++i)
        {
            tmp = rdist(centroid, data_ptr + idx_array_ptr[i] * n_features, n_features);
            radius = Max(radius, tmp);
        }

        node_data_temp[i_node].radius = float(sqrt(radius));
        node_data_temp[i_node].idx_start = idx_start;
        node_data_temp[i_node].idx_end = idx_end;
    };
};

#endif //HDBSCAN_BALLTREE_H

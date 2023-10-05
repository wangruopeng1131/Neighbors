//
// Created by ruopeng on 2022/12/6.
//

#ifndef HDBSCAN_KDTREE_H
#define HDBSCAN_KDTREE_H

#include <cmath>
#include "Eigen/Dense"
#include "BinaryTree.h"

using namespace std;
using namespace Eigen;

class KDTree : public BinaryTree<KDTree>
{
public:
    KDTree(MatrixXd &d, int leaf_size) : BinaryTree<KDTree>(d, leaf_size)
    {
        buildBinaryTree(node_data, 0, 0, n_samples);
    };

    ~KDTree() = default;

    void initNode(deque<NodeData> &node_data_temp, int i_node, int idx_start, int idx_end) noexcept(false)
    {
        double d, rad = 0;
        int i, j;
        unsigned int data_row;

        double *lower_node_ptr = lower_node_bounds_ptr + i_node * n_features;
        double *upper_node_ptr = upper_node_bounds_ptr + i_node * n_features;

        for (i = 0; i < n_features; ++i)
        {
            lower_node_ptr[i] = INFINITY;
            upper_node_ptr[i] = -INFINITY;
        }

        /* Compute the actual data range.  At build time, this is slightly
         slower than using the previously-computed bounds of the parent node,
         but leads to more compact trees and thus faster queries. */

        for (i = idx_start; i < idx_end; ++i)
        {
            data_row = idx_array_ptr[i];
            for (j = 0; j < n_features; ++j)
            {
                d = data_ptr[data_row * n_features + j];
                lower_node_ptr[j] = Min(lower_node_ptr[j], d);
                upper_node_ptr[j] = Max(upper_node_ptr[j], d);
            }
        }

        for (i = 0; i < n_features; ++i)
        {
            rad += Pow(0.5 * Abs(upper_node_ptr[i] - lower_node_ptr[i])); // 默认用欧式距离
        }
        rad = sqrt(rad);
        node_data_temp[i_node].idx_start = idx_start;
        node_data_temp[i_node].idx_end = idx_end;

        /* The radius will hold the size of the circumscribed hypersphere measured
         with the specified metric: in querying, this is used as a measure of the
         size of each node when deciding which nodes to split.*/
        node_data_temp[i_node].radius = (float) rad;
    };
};


#endif //HDBSCAN_KDTREE_H


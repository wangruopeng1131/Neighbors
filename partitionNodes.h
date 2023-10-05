//
// Created by ruopeng on 2022/12/6.
//

#ifndef HDBSCAN_PARTITIONNODES_H
#define HDBSCAN_PARTITIONNODES_H

#include <algorithm>

template<class D, class E>
class IndexComparator
{
private:
    const D &data;
    const double *data_ptr;
    E split_dim, n_features;
public:
    IndexComparator(const D &data, const E &split_dim, const E &n_features) :
            data(data), data_ptr(data.data()), split_dim(split_dim), n_features(n_features)
    {}

    inline bool operator()(const E &a, const E &b) const
    {
        auto a_value = data_ptr[a * n_features + split_dim];
        auto b_value = data_ptr[b * n_features + split_dim];
        return a_value == b_value ? a < b : a_value < b_value;
    }
};

template<class D, class I, class E>
void partition_node_indices_inner(
        const D &data,
        I *node_indices,
        const E &split_dim,
        const E &split_index,
        const E &n_features,
        const E &n_points,
        const E &idx_start)
{
    IndexComparator<D, E> index_comparator(data, split_dim, n_features);

    auto begin = node_indices->begin() + idx_start;
    auto nth = begin + split_index;
    auto end = begin + n_points;
    std::nth_element(
            begin,
            nth,
            end,
            index_comparator);
}

template<class D, class I, class E>
int
partition_node_indices(const D &data, I *node_indices, const E &split_dim, const E &split_index, const E &n_features,
                       const E &n_points, const E &idx_start) noexcept(false)
/*Partition points in the node into two equal-sized groups.
Upon return, the values in node_indices will be rearranged such that
(assuming numpy-style indexing):
data[node_indices[0:split_index], split_dim]
<= data[node_indices[split_index], split_dim]
and
data[node_indices[split_index], split_dim]
<= data[node_indices[split_index:n_points], split_dim]
The algorithm is essentially a partial in-place quicksort around a
set pivot.
Parameters
----------
data : double pointer
Pointer to a 2D array of the training data, of shape [N, n_features].
N must be greater than any of the values in node_indices.
node_indices : int pointer
Pointer to a 1D array of length n_points.  This lists the indices of
        each of the points within the current node.  This will be modified
in-place.
split_dim : int
        the dimension on which to split.  This will usually be computed via
the routine ``find_node_split_dim``.
split_index : int
        the index within node_indices around which to split the points.
n_features: int
        the number of features (i.e columns) in the 2D array pointed by data.
n_points : int
        the length of node_indices. This is also the number of points in
the original dataset.
Returns
-------
status : int
        integer exit status.  On return, the contents of node_indices are
        modified as noted above.
*/
{
partition_node_indices_inner(
        data,
        node_indices,
        split_dim,
        split_index,
        n_features,
        n_points,
        idx_start
);
return 0;
};


#endif //HDBSCAN_PARTITIONNODES_H


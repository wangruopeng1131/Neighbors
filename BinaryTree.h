//
// Created by ruopeng on 2022/12/6.
//

#ifndef HDBSCAN_BINARYTREE_H
#define HDBSCAN_BINARYTREE_H

#include <immintrin.h>
#include <iostream>
#include <type_traits>
#include <memory>
#include <queue>
#include <deque>
#include <numeric>
#include <tuple>
#include <vector>
#include <cmath>
#include <tuple>
#include <omp.h>
#include "Eigen/Dense"
#include "partitionNodes.h"

#define pass (void) 0
#define Max(a, b) ((a) > (b) ? (a) : (b))
#define Min(a, b) ((a) > (b) ? (b) : (a))
#define Abs(a) ((a) > 0 ? (a) : -(a))
#define Pow(a) ((a) * (a))

using namespace std;
using namespace Eigen;

struct alignas(16) NodeHeapData_t
{
    double val = 0.0;
    int i1 = 0;
    int i2 = 0;
};

struct alignas(16) NodeData
{
    int idx_start;
    int idx_end;
    int is_leaf;
    float radius;
};

// 该函数接受两个指向向量的指针 vec1 和 vec2，以及向量的长度 len。使用循环计算两个向量的差的平方和，这里使用了一些 SSE 指令来计算。
//
//首先，我们创建一个 double 型变量的 sum，然后将其初始化为零。接下来，我们使用循环迭代每个双精度浮点数，每次迭代加载两个向量中的两个双精度浮点数并计算其之间的差。然后，我们将差的平方添加到 sum 中，使用 _mm_add_pd 和 _mm_mul_pd SSE 指令。
//
//最后，我们使用 _mm_storeu_pd 指令将 sum 的值存储到 temp 数组中，将其添加到 res 变量中，以便计算向量的差的平方和。我们在循环结尾处理 len 除 2 的余数，并计算向量的差的平方和的平方根。
//
//需要注意的是，该代码适用于长度为偶数的向量，长度为奇数的向量需要进行特殊处理。
//inline double rdist(const double *vec1, const double *vec2, int len) {
//    double t, res = 0.0;
//    for (int i = 0; i < len; ++i) {
//        t = vec1[i] - vec2[i];
//        res += t * t;
//    }
//    return res;
//};

// SSE指令集
inline double rdist(const double *vec1, const double *vec2, int len)
{
    __m128d diff, sum = _mm_setzero_pd();
    int length;
    double res, dif;
    if (len % 2 != 0)
    {
        length = len - 1;
        int i;
        for (i = 0; i < length; i += 2)
        {
            diff = _mm_sub_pd(_mm_load_pd(vec1 + i), _mm_load_pd(vec2 + i));
            sum = _mm_add_pd(sum, _mm_mul_pd(diff, diff));
        }
        double temp[2] = {0.0, 0.0};
        _mm_store_pd(temp, sum);
        res = temp[0] + temp[1];
        dif = vec1[i] - vec2[i];
        res += dif * dif;
    } else
    {
        length = len;
        int i;
        for (i = 0; i < length; i += 2)
        {
            diff = _mm_sub_pd(_mm_load_pd(vec1 + i), _mm_load_pd(vec2 + i));
            sum = _mm_add_pd(sum, _mm_mul_pd(diff, diff));
        }
        double temp[2] = {0.0, 0.0};
        _mm_store_pd(temp, sum);
        res = temp[0] + temp[1];
    }
    return res;
};


//inline double rdist(const double* vec1, const double* vec2, int len) {
//    __m256d v1, v2 ,diff, sum = _mm256_setzero_pd();
//    for (int i = 0; i < len; i += 4) {
//        v1 = _mm256_loadu_pd(vec1 + i);
//        v2 = _mm256_loadu_pd(vec2 + i);
//        diff = _mm256_sub_pd(v1, v2);
//        sum = _mm256_add_pd(sum, _mm256_mul_pd(diff, diff));
//    }
//    double res = 0.0;
//    double temp[4] = {0.0, 0.0, 0.0, 0.0};
//    _mm256_storeu_pd(temp, sum);
//    res += temp[0] + temp[1] + temp[2] + temp[3];
//    for (int i = len - len % 4; i < len; ++i) {
//        double diff = vec1[i] - vec2[i];
//        res += diff * diff;
//    }
//    return res;
//};

inline double dist(double *x1, double *x2, int n_features)
{
    return sqrt(rdist(x1, x2, n_features));
};

class KDTree;

class BallTree;

class NeighborsHeap
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    int n_pts, n_nbrs;
    double *distances_ptr = nullptr;
    int *indices_ptr = nullptr;

    MatrixXd distances;
    MatrixXi indices;


    NeighborsHeap(int n_pts, int n_nbrs) : n_nbrs(n_nbrs), n_pts(n_pts)
    {
        distances.setConstant(n_pts, n_nbrs, INFINITY);
        indices.setZero(n_pts, n_nbrs);
        distances_ptr = distances.data();
        indices_ptr = indices.data();
    };

    ~NeighborsHeap() = default;

    tuple<MatrixXd, MatrixXi> getArray(bool sort = true)
    {
        if (sort)
        {
            _sort();
        }
        tuple<MatrixXd, MatrixXi> res(distances, indices);
        return res;
    };

    [[nodiscard]] inline double largest(int row) const noexcept(false)
    {
        return distances_ptr[row * n_nbrs];
    }

    inline void _sort() noexcept(false)
    {

        distances_ptr = distances.data();
        indices_ptr = indices.data();
        MatrixXd returnDis;
        MatrixXi returnInd;
        returnDis.setZero(distances.rows(), distances.cols());
        returnInd.setZero(distances.rows(), distances.cols());
        double *returnDis_ptr = returnDis.data();
        int *returnInd_ptr = returnInd.data();
        int rows = int(distances.rows());
        int cols = int(distances.cols());

        std::vector<int> idx(cols);
#pragma omp parallel firstprivate(idx) shared(returnDis, returnInd)
        {
#pragma omp for schedule(static)
            for (int i = 0; i < rows; ++i)
            {
                std::iota(idx.begin(), idx.end(), 0);
                double *inArray = distances_ptr + i * cols;
                const auto function = [inArray](int i1, int i2) noexcept -> bool
                {
                    return inArray[i1] < inArray[i2];
                };
                std::stable_sort(idx.begin(), idx.end(), function);

                for (int j = 0; j < cols; ++j)
                {
                    returnDis_ptr[i * cols + j] = distances_ptr[i * cols + idx[j]];
                    returnInd_ptr[i * cols + j] = indices_ptr[i * cols + idx[j]];
                }
            }
        }
        distances = returnDis;
        indices = returnInd;
    };


    [[nodiscard]] int push(int row, double val, int i_val) const noexcept(false)
/*Push a tuple (val, val_idx) onto a fixed-size max-heap.
The max-heap is represented as a Structure of Arrays where:
- values is the array containing the data to construct the heap with
- indices is the array containing the indices (meta-data) of each value
Notes
-----
        Arrays are manipulated via a pointer to there first element and their size
as to ease the processing of dynamically allocated buffers.
        For instance, in pseudo-code:
values = [1.2, 0.4, 0.1],
indices = [42, 1, 5],
heap_push(
        values=values,
        indices=indices,
        size=3,
        val=0.2,
        val_idx=4,
)
will modify values and indices inplace, giving at the end of the call:
values  == [0.4, 0.2, 0.1]
indices == [1, 4, 5]*/
    {
        /*Check if val should be in heap*/
        if (val >= distances_ptr[row * n_nbrs])
        {
            return 0;
        }

        /*Insert val at position zero*/
        distances_ptr[row * n_nbrs] = val;
        indices_ptr[row * n_nbrs] = i_val;

        /*Descend the heap, swapping values until the max heap criterion is met*/
        int current_idx = 0;
        int swap_idx, left_child_idx, right_child_idx;
        while (true)
        {
            left_child_idx = 2 * current_idx + 1;
            right_child_idx = left_child_idx + 1;

            if (left_child_idx >= n_nbrs)
            {
                break;
            } else if (right_child_idx >= n_nbrs)
            {
                if (distances_ptr[row * n_nbrs + left_child_idx] > val)
                {
                    swap_idx = left_child_idx;
                } else
                {
                    break;
                }
            } else if (distances_ptr[row * n_nbrs + left_child_idx] >= distances_ptr[row * n_nbrs + right_child_idx])
            {
                if (val < distances_ptr[row * n_nbrs + left_child_idx])
                {
                    swap_idx = left_child_idx;
                } else
                {
                    break;
                }
            } else
            {
                if (val < distances_ptr[row * n_nbrs + right_child_idx])
                {
                    swap_idx = right_child_idx;
                } else
                {
                    break;
                }
            }
            distances_ptr[row * n_nbrs + current_idx] = distances_ptr[row * n_nbrs + swap_idx];
            indices_ptr[row * n_nbrs + current_idx] = indices_ptr[row * n_nbrs + swap_idx];
            current_idx = swap_idx;
        }
        distances_ptr[row * n_nbrs + current_idx] = val;
        indices_ptr[row * n_nbrs + current_idx] = i_val;
        return 0;
    };
};

class NodeHeap
{
    /*NodeHeap
    This is a min-heap implementation for keeping track of nodes
    during a breadth-first search.  Unlike the NeighborsHeap above,
            the NodeHeap does not have a fixed size and must be able to grow
            as elements are added.
    Internally, the data is stored in a simple binary heap which meets
            the min heap condition:
            heap[i].val < min(heap[2 * i + 1].val, heap[2 * i + 2].val)*/
public:
    vector<NodeHeapData_t> data;
    NodeHeapData_t *data_val_ptr = nullptr;
    int n;

    explicit NodeHeap(int size_guess = 100)
    {
        size_guess = max(size_guess, 1);
        data = vector<NodeHeapData_t>(size_guess);
        n = size_guess;
        clear();
    };

    ~NodeHeap() = default;

    int resize(int new_size) noexcept(false)
    {
        int size = data.size();
        vector<NodeHeapData_t> new_data(new_size);

        if (size > 0 && new_size > 0)
        {
            for (int i = 0; i < min(size, new_size); ++i)
            {
                new_data[i] = data[i];
            }
        }

        if (new_size < size)
        {
            n = new_size;
        }
        data = new_data;
        return 0;
    };

    int push(NodeHeapData_t d) noexcept(false)
    /*Push a new item onto the heap*/
    {
        n += 1;
        if (n > data.size())
        {
            resize(2 * n);
        }

        NodeHeapData_t temp;
        int i = n - 1;
        data[i] = d;
        while (i > 0)
        {
            int i_parent = (i - 1) / 2;
            if (data[i_parent].val <= data[i].val)
            {
                break;
            } else
            {
                temp = data[i];
                data[i] = data[i_parent];
                data[i_parent] = temp;
                i = i_parent;
            }
        }
        return 0;
    };

    NodeHeapData_t peek()
    {
        return data[0];
    };

    NodeHeapData_t pop()
    /*Remove the root of the heap, and update the remaining nodes*/
    {
        if (n == 0)
        {
            cerr << "cannot pop on empty heap" << endl;
        }
        data_val_ptr = data.data();
        NodeHeapData_t popped_element = data_val_ptr[0];

        /*# pop off the first element, move the last element to the front,
         and then perform swaps until the heap is back in order*/

        data_val_ptr[0] = data_val_ptr[n - 1];

        n -= 1;
        int i_child1, i_child2, i_swap;
        int i = 0;
        int tempN = n;
        NodeHeapData_t temp;

        while (i < tempN)
        {
            i_child1 = 2 * i + 1;
            i_child2 = 2 * i + 2;
            i_swap = 0;
            if (i_child2 < tempN)
            {
                if (data_val_ptr[i_child1].val <= data_val_ptr[i_child2].val)
                {
                    i_swap = i_child1;
                } else
                {
                    i_swap = i_child2;
                }
            } else if (i_child1 < tempN)
            {
                i_swap = i_child1;
            } else
            {
                break;
            }

            if (i_swap > 0 && data_val_ptr[i_swap].val <= data_val_ptr[i].val)
            {
                temp = data_val_ptr[i];
                data_val_ptr[i] = data_val_ptr[i_swap];
                data_val_ptr[i] = temp;
                i = i_swap;
            } else
            {
                break;
            }
        }
        return popped_element;
    };

    void clear()
    {
        n = 0;
    };
};

template<typename Tree>
class BinaryTree
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    MatrixXd data;
    VectorXi idx_array;
    MatrixXd lower_node_bounds;
    MatrixXd upper_node_bounds;

    deque<NodeData> node_data;
    int n_samples, n_features;

    int leaf_sizes;
    int n_levels;
    int n_nodes;
    int n_trims{};
    int n_leaves{};
    int n_splits{};

    double *data_ptr = nullptr;
    double *lower_node_bounds_ptr = nullptr;
    double *upper_node_bounds_ptr = nullptr;
    int *idx_array_ptr = nullptr;
    double *node1_lower_bounds = nullptr;
    double *node1_upper_bounds = nullptr;
    double *node2_lower_bounds = nullptr;
    double *node2_upper_bounds = nullptr;

    Tree *derivedTree = nullptr;
    NodeData *self_node_ptr = nullptr;
    NodeData *other_node_ptr = nullptr;

    BinaryTree(MatrixXd &d, int leaf_size) : data(d), leaf_sizes(leaf_size)
    {
        n_samples = int(d.rows());
        n_features = int(d.cols());

        /* determine number of levels in the tree, and from this
         the number of nodes in the tree.  This results in leaf nodes
         with numbers of points between leaf_size and 2 * leaf_size */
        n_levels = log2(fmax(1, (float) (n_samples - 1) / (float) leaf_size)) + 1;
        n_nodes = (2 << (n_levels - 1)) - 1;

        /*allocate arrays for storage*/
        idx_array.setLinSpaced(n_samples, 0, n_samples);
        node_data = deque<NodeData>(n_nodes);
        lower_node_bounds.setZero(n_nodes, n_features);
        upper_node_bounds.setZero(n_nodes, n_features);

        upper_node_bounds_ptr = upper_node_bounds.data();
        lower_node_bounds_ptr = lower_node_bounds.data();
        idx_array_ptr = idx_array.data();
        data_ptr = data.data();

        derivedTree = static_cast<Tree *>(this);
    };


    ~BinaryTree() = default;


    void buildBinaryTree(deque<NodeData> &node_data_temp, const int i_node, const int idx_start, const int idx_end)
    {
        //样本的数量
        int i_max;
        int n_points = idx_end - idx_start;
        int n_mid = n_points / 2;

        /*initialize node data*/
        derivedTree->initNode(node_data_temp, i_node, idx_start, idx_end);

        if (2 * i_node + 1 >= n_nodes)
        {
            node_data_temp[i_node].is_leaf = true;
            if (idx_end - idx_start > 2 * leaf_sizes)
            {
                cerr << "Internal: memory layout is flawed: not enough nodes allocated" << endl;
            }
        } else if (idx_end - idx_start < 2)
        {
            node_data_temp[i_node].is_leaf = true;
            cerr << "Internal: memory layout is flawed: too many nodes allocated" << endl;

        } else
        {
            /*split node and recursively construct child nodes.*/
            node_data_temp[i_node].is_leaf = false;
            i_max = findNodeSplitDim(idx_start, idx_end);

            partition_node_indices<MatrixXd, VectorXi, int>(data, &idx_array, i_max, n_mid, n_features, n_points,
                                                            idx_start);
            buildBinaryTree(node_data_temp, 2 * i_node + 1, idx_start, idx_start + n_mid);
            buildBinaryTree(node_data_temp, 2 * i_node + 2, idx_start + n_mid, idx_end);
        }
    };//构建kd树


    inline int findNodeSplitDim(int idx_start, int idx_end)
    {
        double min_val, max_val, val, spread, max_spread = 0;
        int j_max = 0;

        for (int j = 0; j < n_features; ++j)
        {
            max_val = data_ptr[idx_array_ptr[idx_start] * n_features + j];
            min_val = max_val;
            for (int i = 1; i < idx_end - idx_start; ++i)
            {
                val = data_ptr[idx_array_ptr[idx_start + i] * n_features + j];
                max_val = Max(max_val, val);
                min_val = Min(min_val, val);
            }
            spread = max_val - min_val;
            if (spread > max_spread)
            {
                max_spread = spread;
                j_max = j;
            }
        }
        return j_max;
    };

    int
    queryRadius(MatrixXd &X, double r, vector<vector<int>> &returnIdx, vector<vector<double>> &returnDist,
                vector<int> &returnCount) noexcept(false)
    /*Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    An array of points to query
    r : distance within which neighbors are returned
    r can be a single value, or an array of values of shape
    x.shape[:-1] if different radii are desired for each point.
    return_distance : bool, default=False
    if True,  return distances to neighbors of each point
    if False, return only neighbors
    Note that unlike the query() method, setting return_distance=True
    here adds to the computation time.  Not all distances need to be
    calculated explicitly for return_distance=False.  Results are
    not sorted by default: see ``sort_results`` keyword.
    count_only : bool, default=False
    if True,  return only the count of points within distance r
    if False, return the indices of all points within distance r
    If return_distance==True, setting count_only=True will
            result in an error.
    sort_results : bool, default=False
    if True, the distances and indices will be sorted before being
    returned.  If False, the results will not be sorted.  If
            return_distance == False, setting sort_results = True will
            result in an error.
    Returns
    -------
    count       : if count_only == True
            ind         : if count_only == False and return_distance == False
    (ind, dist) : if count_only == False and return_distance == True
            count : ndarray of shape X.shape[:-1], dtype=int
    Each entry gives the number of neighbors within a distance r of the
            corresponding point.
    ind : ndarray of shape X.shape[:-1], dtype=object
    Each element is a numpy integer array listing the indices of
            neighbors of the corresponding point.  Note that unlike
            the results of a k-neighbors query, the returned neighbors
    are not sorted by distance by default.
    dist : ndarray of shape X.shape[:-1], dtype=object
    Each element is a numpy double array listing the distances
            corresponding to indices in i.*/
    {
        int size = int(X.rows());
        double *pt = X.data();


        node1_lower_bounds = lower_node_bounds.data();
        node1_upper_bounds = upper_node_bounds.data();

        returnIdx.resize(size);
        returnDist.resize(size);
        returnCount.resize(size);
//#pragma omp parallel for num_threads(omp_get_num_threads())
        for (int i = 0; i < size; ++i)
        {
            returnIdx[i].resize(size);
            returnDist[i].resize(size);
            int count = queryRadiusSingle(0, pt + i * n_features, r, returnIdx[i].data(), returnDist[i].data(), 0);
            returnCount[i] = count;
            returnIdx[i].resize(count);
            returnDist[i].resize(count);
//            memcpy_s(returnIdx[i].data(), count * sizeof(int), indices.data(), count * sizeof(int));
//            memcpy_s(returnDist[i].data(), count * sizeof(double), distances.data(), count * sizeof(double));
        }
        return 0;
    }

    int queryRadiusSingle(int i_node, double *pt, double r, int *indices, double *distances, int count) noexcept(false)
    /*recursive single-tree radius query, depth-first*/
    {
        NodeData node_info = node_data[i_node];
        int i;
        double reduced_r, dist_pt, dist_LB = 0, dist_UB = 0;

        minMaxDist(derivedTree, i_node, pt, dist_LB, dist_UB);

        //  Case 1: all node points are outside distance r. prune this branch.
        if (dist_LB > r)
        {
            pass;
        }
            // Case 2: all node points are within distance r add all points to neighbors
        else if (dist_UB <= r)
        {
//            count += node_info.idx_end - node_info.idx_start;
            for (i = node_info.idx_start; i < node_info.idx_end; ++i)
            {
                if (count < 0 || count >= n_samples)
                {
                    return -1;
                }
                indices[count] = idx_array_ptr[i];
                distances[count] = dist(pt, data_ptr + idx_array_ptr[i] * n_features, n_features);
                count += 1;
            }
        }
            // Case 3: this is a leaf node.  Go through all points to determine if they fall within radius
        else if (node_info.is_leaf)
        {
            reduced_r = r * r;

            for (i = node_info.idx_start; i < node_info.idx_end; ++i)
            {
                dist_pt = rdist(pt, data_ptr + idx_array_ptr[i] * n_features, n_features);
                if (dist_pt <= reduced_r)
                {
                    if (count < 0 || count >= n_samples)
                    {
                        return -1;
                    } else
                    {
                        indices[count] = idx_array_ptr[i];
                        distances[count] = sqrt(dist_pt);
                    }
                    count += 1;
                }
            }
        }
            // Case 4: Node is not a leaf.  Recursively query subnodes
        else
        {
            count = queryRadiusSingle(2 * i_node + 1, pt, r, indices, distances, count);
            count = queryRadiusSingle(2 * i_node + 2, pt, r, indices, distances, count);
        }
        return count;
    }

    auto query(MatrixXd &X, int k = 1, bool dualTree = false, bool breadthFirst = false) noexcept(false)
    /*query(X, k=1, returnDistance=True,
            dualTree=False, breadthFirst=False)
    query the tree for the k nearest neighbors
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    An array of points to query
    k : int, default=1
    The number of nearest neighbors to return
    return_distance : bool, default=True
    if True, return a tuple (d, i) of distances and indices
    if False, return array i
    dualtree : bool, default=False
    if True, use the dual tree formalism for the query: a tree is
            built for the query points, and the pair of trees is used to
            efficiently search this space.  This can lead to better
            performance as the number of points grows large.
    breadth_first : bool, default=False
    if True, then query the nodes in a breadth-first manner.
    Otherwise, query the nodes in a depth-first manner.
    sort_results : bool, default=True
    if True, then distances and indices of each point are sorted
    on return, so that the first column contains the closest points.
    Otherwise, neighbors are returned in an arbitrary order.
    Returns
    -------
    i    : if return_distance == False
    (d,i) : if return_distance == True
            d : ndarray of shape X.shape[:-1] + (k,), dtype=double
    Each entry gives the list of distances to the neighbors of the
    corresponding point.
    i : ndarray of shape X.shape[:-1] + (k,), dtype=int
    Each entry gives the list of indices of neighbors of the
            corresponding point.*/
    {
        if (X.cols() != n_features)
        {
            cerr << "query data dimension must match training data dimension" << endl;
        }
        if (n_samples < k)
        {
            cerr << "k must be less than or equal to the number of training points" << endl;
        }

        NeighborsHeap heap = NeighborsHeap(int(X.rows()), k);
        NodeHeap nodeHeap;
        if (breadthFirst)
        {
            nodeHeap = NodeHeap(n_samples / leaf_sizes);
        }

        n_trims = 0;
        n_leaves = 0;
        n_splits = 0;


        if (dualTree)
        {
            Tree other = Tree(X, leaf_sizes);
            node1_lower_bounds = lower_node_bounds.data();
            node1_upper_bounds = upper_node_bounds.data();
            node2_lower_bounds = other.lower_node_bounds.data();
            node2_upper_bounds = other.upper_node_bounds.data();

            vector<NodeData> selfNode(node_data.begin(), node_data.end());
            vector<NodeData> otherNode(other.node_data.begin(), other.node_data.end());

            self_node_ptr = selfNode.data();
            other_node_ptr = otherNode.data();

            if (breadthFirst)
            {
                queryDualBreadthFirst(&other, &heap, &nodeHeap);
            } else
            {
                double reduced_dist_LB = minRdistDual(derivedTree, 0, &other, 0);
                VectorXd bounds = VectorXd::Constant(other.node_data.size(), INFINITY);
                queryDualDepthFirst(0, &other, 0, bounds.data(), &heap, reduced_dist_LB);
            }
        } else
        {
            double *pt = X.data();
            data_ptr = data.data();
            idx_array_ptr = idx_array.data();
            int rows = int(X.rows());
            node1_lower_bounds = lower_node_bounds.data();
            node1_upper_bounds = upper_node_bounds.data();
            vector<NodeData> selfNode(node_data.begin(), node_data.end());
            self_node_ptr = selfNode.data();

            if (breadthFirst)
            {
#pragma omp parallel for num_threads(omp_get_num_threads())
                for (int i = 0; i < rows; ++i)
                {
                    querySingleBreadthfirst(i, self_node_ptr, &heap, &nodeHeap, pt + i * n_features);
                }
            } else
            {
#pragma omp parallel for num_threads(omp_get_num_threads())
                for (int i = 0; i < rows; ++i)
                {
                    double reduced_dist_LB = minRdist(derivedTree, 0, pt + i * n_features);
                    querySingleDepthfirst(i, 0, reduced_dist_LB, self_node_ptr, &heap, &nodeHeap,
                                          pt + i * n_features);
                }
            }
        }
        auto res = heap.getArray(true);
        return res;
    };

    int querySingleBreadthfirst(int i_pt, NodeData *node_data_ptr, NeighborsHeap *heap, NodeHeap *nodeHeap,
                                double *pt) noexcept(false)
    {
        int i, i_node;
        double dist_pt, reduced_dist_LB;
        NodeHeapData_t nodeheap_item{minRdist(derivedTree, 0, pt), 0};
        NodeData node_info{};
        nodeHeap->push(nodeheap_item);

        while (nodeHeap->n > 0)
        {
            nodeheap_item = nodeHeap->pop();
            reduced_dist_LB = nodeheap_item.val;
            i_node = nodeheap_item.i1;
            node_info = node_data_ptr[i_node];

            // Case 1: query point is outside node radius:trim it from the query
            if (reduced_dist_LB > heap->largest(i_pt))
            {
                n_trims += 1;
            }
                // Case 2: this is a leaf node.  Update set of nearby points
            else if (node_info.is_leaf)
            {
                n_leaves += 1;
                for (i = node_info.idx_start; i < node_info.idx_end; ++i)
                {
                    dist_pt = rdist(pt, data_ptr + idx_array_ptr[i] * n_features, n_features);
                    heap->push(i_pt, dist_pt, idx_array_ptr[i]);
                }
            }
                // Case 3: Node is not a leaf.  Add subnodes to the node heap
            else
            {
                n_splits += 1;
                for (i = 2 * i_node + 1; i < 2 * i_node + 3; ++i)
                {
                    nodeheap_item.i1 = i;
                    nodeheap_item.val = minRdist(derivedTree, i, pt);
                    nodeHeap->push(nodeheap_item);
                }
            }
        }
        return 0;
    };

    int querySingleDepthfirst(int i_pt, int i_node, double reduced_dist_LB, NodeData *node_data_ptr,
                              NeighborsHeap *heap, NodeHeap *nodeHeap,
                              double *pt) noexcept(false)
    {
        NodeData node_info = node_data_ptr[i_node];
        double dist_pt, reduced_dist_LB_1, reduced_dist_LB_2;
        int i, i1, i2;

        // Case 1: query point is outside node radius:trim it from the query
        if (reduced_dist_LB > heap->largest(i_pt))
        {
            n_trims += 1;
        }
            // Case 2: this is a leaf node.  Update set of nearby points
        else if (node_info.is_leaf)
        {
            n_leaves += 1;
            for (i = node_info.idx_start; i < node_info.idx_end; ++i)
            {
                dist_pt = rdist(pt, data_ptr + idx_array_ptr[i] * n_features, n_features);
                heap->push(i_pt, dist_pt, idx_array_ptr[i]);
            }
        }
            // Case 3: Node is not a leaf.  Recursively query subnodes starting with the closest
        else
        {
            n_splits += 1;
            i1 = 2 * i_node + 1;
            i2 = i1 + 1;
            reduced_dist_LB_1 = minRdist(derivedTree, i1, pt);
            reduced_dist_LB_2 = minRdist(derivedTree, i2, pt);

            if (reduced_dist_LB_1 < reduced_dist_LB_2)
            {
                querySingleDepthfirst(i_pt, i1, reduced_dist_LB_1, node_data_ptr, heap, nodeHeap, pt);
                querySingleDepthfirst(i_pt, i2, reduced_dist_LB_2, node_data_ptr, heap, nodeHeap, pt);
            } else
            {
                querySingleDepthfirst(i_pt, i2, reduced_dist_LB_2, node_data_ptr, heap, nodeHeap, pt);
                querySingleDepthfirst(i_pt, i1, reduced_dist_LB_1, node_data_ptr, heap, nodeHeap, pt);
            }
        }
        return 0;

    };

    int queryDualBreadthFirst(Tree *tree, NeighborsHeap *heap, NodeHeap *nodeHeap) noexcept(false)
    /*Non-recursive dual-tree k-neighbors query, breadth-first*/
    {
        // Set up the node heap and push the head nodes onto it
        NodeHeapData_t first = NodeHeapData_t{minRdistDual(this, 0, tree, 0), 0, 0};
        nodeHeap->push(first);
        int i_node1, i_node2, i1, i2, i_pt;
        double dist_pt, reduced_dist_LB;

        NodeData node_info1{}, node_info2{};
        VectorXd bounds = VectorXd::Constant(tree->node_data.size(), INFINITY);
        double *bounds_ptr = bounds.data();
        double *treeData_ptr = tree->data.data();
        int *treeIdx_ptr = tree->idx_array.data();
        NodeHeapData_t nodeheap_item{};
        while (nodeHeap->n > 0)
        {
            nodeheap_item = nodeHeap->pop();
            reduced_dist_LB = nodeheap_item.val;
            i_node1 = nodeheap_item.i1;
            i_node2 = nodeheap_item.i2;

            node_info1 = self_node_ptr[i_node1];
            node_info2 = other_node_ptr[i_node2];

            /*Case 1: nodes are further apart than the current bound: trim both from the query*/
            if (reduced_dist_LB > bounds_ptr[i_node2])
            {
                pass;
            }

                /*Case 2: both nodes are leaves:do a brute-force search comparing all pairs*/
            else if (node_info1.is_leaf & node_info2.is_leaf)
            {
                bounds_ptr[i_node2] = -1;
                for (i2 = node_info2.idx_start; i2 < node_info2.idx_end; ++i2)
                {
                    i_pt = treeIdx_ptr[i2];

                    if (heap->largest(i_pt) <= reduced_dist_LB)
                    {
                        continue;
                    }

                    for (i1 = node_info1.idx_start; i1 < node_info1.idx_end; ++i1)
                    {

                        dist_pt = rdist(data_ptr + idx_array_ptr[i1] * n_features, treeData_ptr + i_pt * n_features,
                                        n_features);
                        heap->push(i_pt, dist_pt, idx_array_ptr[i1]);
                    }
                    /*keep track of node bound*/
                    bounds_ptr[i_node2] = Max(bounds_ptr[i_node2], heap->largest(i_pt));
                }
            }
                /*Case 3a: node 1 is a leaf or is smaller: split node 2 and recursively query, starting with the nearest subnode*/
            else if (node_info1.is_leaf | (!node_info2.is_leaf && node_info2.radius > node_info1.radius))
            {
                nodeheap_item.i1 = i_node1;
                for (i2 = 2 * i_node2 + 1; i2 < 2 * i_node2 + 3; ++i2)
                {
                    nodeheap_item.val = minRdistDual(derivedTree, i_node1, tree, i2);
                    nodeheap_item.i2 = i2;
                    nodeHeap->push(nodeheap_item);
                }

            }
                /*Case 3b: node 2 is a leaf or is smaller: split node 1 and recursively query, starting with the nearest subnode*/
            else
            {
                nodeheap_item.i2 = i_node2;
                for (i1 = 2 * i_node1 + 1; i1 < 2 * i_node1 + 3; ++i1)
                {
                    nodeheap_item.val = minRdistDual(derivedTree, i1, tree, i_node2);
                    nodeheap_item.i1 = i1;
                    nodeHeap->push(nodeheap_item);
                }
            }
        }
        return 0;
    };


    int queryDualDepthFirst(int node1, Tree *tree, int node2, double *bounds_ptr, NeighborsHeap *heap,
                            double reduced_dist_LB) noexcept(false)
    /*Recursive dual-tree k-neighbors query, depth-first*/
    {
        NodeData node_info1 = self_node_ptr[node1];
        NodeData node_info2 = other_node_ptr[node2];
        double *treeData_ptr = tree->data.data();
        int *tree_idx_array_ptr = tree->idx_array.data();
        double bound_max, dist_pt, reduced_dist_LB1, reduced_dist_LB2;
        int i1, i2, i_pt, i_parent;

        //Case 1: nodes are further apart than the current bound:trim both from the query
        if (reduced_dist_LB > bounds_ptr[node2])
        {
            pass;
        }

            //Case 2: both nodes are leaves:do a brute-force search comparing all pairs
        else if (node_info1.is_leaf & node_info2.is_leaf)
        {
            bounds_ptr[node2] = 0;

            for (i2 = node_info2.idx_start; i2 < node_info2.idx_end; ++i2)
            {
                i_pt = tree_idx_array_ptr[i2];

                if (heap->largest(i_pt) <= reduced_dist_LB)
                {
                    continue;
                }

                for (i1 = node_info1.idx_start; i1 < node_info1.idx_end; ++i1)
                {
                    dist_pt = rdist(data_ptr + idx_array_ptr[i1] * n_features, treeData_ptr + i_pt * n_features,
                                    n_features);
                    heap->push(i_pt, dist_pt, idx_array_ptr[i1]);
                }

                bounds_ptr[node2] = Max(bounds_ptr[node2], heap->largest(i_pt));
            }

            while (node2 > 0)
            {
                i_parent = (node2 - 1) / 2;
                bound_max = Max(bounds_ptr[2 * i_parent + 1], bounds_ptr[2 * i_parent + 2]);

                if (bound_max < bounds_ptr[i_parent])
                {
                    bounds_ptr[i_parent] = bound_max;
                    node2 = i_parent;
                } else
                {
                    break;
                }
            }
        }
            //Case 3a: node 1 is a leaf or is smaller: split node 2 and recursively query, starting with the nearest subnode
        else if (node_info1.is_leaf || (!node_info2.is_leaf && node_info2.radius > node_info1.radius))
        {
            reduced_dist_LB1 = minRdistDual(this, node1, tree, 2 * node2 + 1);
            reduced_dist_LB2 = minRdistDual(this, node1, tree, 2 * node2 + 2);

            if (reduced_dist_LB1 < reduced_dist_LB2)
            {
                queryDualDepthFirst(node1, tree, 2 * node2 + 1, bounds_ptr, heap, reduced_dist_LB1);
                queryDualDepthFirst(node1, tree, 2 * node2 + 2, bounds_ptr, heap, reduced_dist_LB2);
            } else
            {
                queryDualDepthFirst(node1, tree, 2 * node2 + 2, bounds_ptr, heap, reduced_dist_LB2);
                queryDualDepthFirst(node1, tree, 2 * node2 + 1, bounds_ptr, heap, reduced_dist_LB1);
            }
        }
            // Case 3b: node 2 is a leaf or is smaller: split node 1 and recursively query, starting with the nearest subnode
        else
        {
            reduced_dist_LB1 = minRdistDual(this, 2 * node1 + 1, tree, node2);
            reduced_dist_LB2 = minRdistDual(this, 2 * node1 + 2, tree, node2);

            if (reduced_dist_LB1 < reduced_dist_LB2)
            {
                queryDualDepthFirst(2 * node1 + 1, tree, node2, bounds_ptr, heap, reduced_dist_LB1);
                queryDualDepthFirst(2 * node1 + 2, tree, node2, bounds_ptr, heap, reduced_dist_LB2);
            } else
            {
                queryDualDepthFirst(2 * node1 + 2, tree, node2, bounds_ptr, heap, reduced_dist_LB2);
                queryDualDepthFirst(2 * node1 + 1, tree, node2, bounds_ptr, heap, reduced_dist_LB1);
            }
        }
        return 0;
    };


    inline double
    minRdistDual(BinaryTree *derived, int i_node1, KDTree *tree2, int i_node2) noexcept(false)
    /*Compute the minimum reduced distance between two nodes*/
    {
        double d, d1, d2, rdist = 0.0;
        double *node1_lower_bounds_ptr = node1_lower_bounds + i_node1 * n_features;
        double *node1_upper_bounds_ptr = node1_upper_bounds + i_node1 * n_features;
        double *node2_lower_bounds_ptr = node2_lower_bounds + i_node2 * n_features;
        double *node2_upper_bounds_ptr = node2_upper_bounds + i_node2 * n_features;
        /* here we'll use the fact that x + abs(x) = 2 * max(x, 0) */
        for (int i = 0; i < n_features; ++i)
        {
            d1 = node1_lower_bounds_ptr[i] - node2_upper_bounds_ptr[i];
            d2 = node2_lower_bounds_ptr[i] - node1_upper_bounds_ptr[i];
            d = (d1 + Abs(d1)) + (d2 + Abs(d2));

            rdist += Pow(0.5 * d);
        }
        return rdist;
    };

    inline double minRdist(KDTree *derived, int i_node, double *pt)
    /*Compute the minimum reduced-distance between a point and a node*/
    {
        double d, d_lo, d_hi, rdist = 0.0;
        double *node1_lower_bounds_ptr = node1_lower_bounds + i_node * n_features;
        double *node1_upper_bounds_ptr = node1_upper_bounds + i_node * n_features;
        for (int j = 0; j < n_features; ++j)
        {
            d_lo = node1_lower_bounds_ptr[j] - pt[j];
            d_hi = pt[j] - node1_upper_bounds_ptr[j];
            d = (d_lo + Abs(d_lo)) + (d_hi + Abs(d_hi));
            rdist += Pow(0.5 * d);
        }
        return rdist;
    };

    inline int
    minMaxDist(KDTree *derived, int i_node, double *pt, double &min_dist, double &max_dist) noexcept(false)
    {
        double d, d_lo, d_hi, temp;
        double *node1_lower_bounds_ptr = node1_lower_bounds + i_node * n_features;
        double *node1_upper_bounds_ptr = node1_upper_bounds + i_node * n_features;;
        for (int i = 0; i < n_features; ++i)
        {
            d_lo = node1_lower_bounds_ptr[i] - pt[i];
            d_hi = pt[i] - node1_upper_bounds_ptr[i];
            d = (d_lo + Abs(d_lo)) + (d_hi + Abs(d_hi));
            temp = Max(Abs(d_lo), Abs(d_hi));
            min_dist += Pow(0.5 * d);
            max_dist += Pow(temp);
        }

        min_dist = sqrt(min_dist);
        max_dist = sqrt(max_dist);
        return 0;
    };

    inline double
    minRdistDual(BinaryTree *tree1, int i_node1, BallTree *tree2, int i_node2) noexcept(false)
    {
        double *node1_lower_bounds_ptr = node1_lower_bounds + i_node1 * n_features;
        double *node2_lower_bounds_ptr = node2_lower_bounds + i_node2 * n_features;
        double dist_pt = dist(node1_lower_bounds_ptr, node2_lower_bounds_ptr, n_features);
        dist_pt = Max(0, dist_pt - self_node_ptr[i_node1].radius - other_node_ptr[i_node2].radius);
        return dist_pt * dist_pt;
    };


    inline double minRdist(BallTree *tree, int i_node, double *pt) noexcept(false)
    {
        double *node1_lower_bounds_ptr = node1_lower_bounds + i_node * n_features;
        double dist_pt = dist(pt, node1_lower_bounds_ptr, n_features);
        dist_pt = Max(0, dist_pt - self_node_ptr[i_node].radius);
        return dist_pt * dist_pt;
    };

    inline int
    minMaxDist(BallTree *derived, int i_node, double *pt, double &min_dist, double &max_dist) noexcept(false)
    {
        double *node1_lower_bounds_ptr = node1_lower_bounds + i_node * n_features;
        double dist_pt = dist(pt, node1_lower_bounds_ptr, n_features);

        double rad = self_node_ptr[i_node].radius;
        min_dist = Max(dist_pt - rad, 0);
        max_dist = dist_pt + rad;
        return 0;
    };

    MatrixXd &getData()
    {
        return data;
    }

    [[nodiscard]] int numFeatures() const
    {
        return n_features;
    }
};


#endif //HDBSCAN_BINARYTREE_H

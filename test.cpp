//
// Created by wangr on 2023/6/8.
//
#include <intrin.h>
#include <ctime>
#include <iostream>

#define SIZE 100000

inline double rdist_128(const double *vec1, const double *vec2, int len)
{
    __m128d diff, sum = _mm_setzero_pd();
    for (int i = 0; i < len; i += 2)
    {
        diff = _mm_sub_pd(_mm_load_pd(vec1 + i), _mm_load_pd(vec2 + i));
        sum = _mm_add_pd(sum, _mm_mul_pd(diff, diff));
    }
    double temp[2] = {0.0, 0.0};
    _mm_store_pd(temp, sum);
    double dif, res = temp[0] + temp[1];
    for (int i = len - len % 2; i < len; ++i)
    {
        dif = vec1[i] - vec2[i];
        res += dif * dif;
    }
    return res;
};

inline double rdist_256(const double *vec1, const double *vec2, int len)
{
    __m256d diff, sum = _mm256_setzero_pd();
    for (int i = 0; i < len; i += 4)
    {
        diff = _mm256_sub_pd(_mm256_load_pd(vec1 + i), _mm256_load_pd(vec2 + i));
        sum = _mm256_add_pd(sum, _mm256_mul_pd(diff, diff));
    }
    double temp[4] = {0.0, 0.0, 0.0, 0.0};
    _mm256_store_pd(temp, sum);
    double dif, res = temp[0] + temp[1] + temp[2] + temp[3] ;
    for (int i = len - len % 4; i < len; ++i)
    {
        dif = vec1[i] - vec2[i];
        res += dif * dif;
    }
    return res;
};

int main()
{
    using align_double_16 = __declspec (align(16)) double;
    using align_double_32 = __declspec (align(32)) double;

    clock_t start, end;

    align_double_16 a[13] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

    align_double_16 b[13] = {11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, -1};

    align_double_32 c[13] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

    align_double_32 d[13] = {11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, -1};

    start = clock();
    double res;
    for (int i = 0; i <SIZE; ++i)
    {
        res = rdist_128(a, b, 13);
    }
    end = clock();
    std::cout << "result:" << res << std::endl;
    std::cout << "run time:" << (double) (end - start) / CLOCKS_PER_SEC << "S" << std::endl;

    start = clock();
    for (int i = 0; i < SIZE; ++i)
    {
        res = rdist_256(c, d, 13);
    }
    end = clock();
    std::cout << "result:" << res << std::endl;
    std::cout << "run time:" << (double) (end - start) / CLOCKS_PER_SEC << "S" << std::endl;

    __declspec (align(16)) float A[SIZE], B[SIZE], C[SIZE]; // GCC的内存对齐
    start = clock();
    for (int i = 0; i < SIZE; i += 4)
    {
        _mm_store_ps(C + i,  _mm_add_ps(_mm_load_ps(A + i), _mm_load_ps(B + i))); // 用store和load替换storeu和loadu
    }
    end = clock();
    std::cout << "run time:" << (double) (end - start) / CLOCKS_PER_SEC << "S" << std::endl;

    __declspec (align(32)) float E[SIZE], F[SIZE], G[SIZE]; // 32字节对齐
    start = clock();
    for (int i = 0; i < SIZE; i += 8) // 循环跨度修改为8
    {
        *(__m256 *)(E + i) = _mm256_add_ps(*(__m256 *)(F + i), *(__m256 *)(G + i)); // 使用256位宽的数据与函数
    }
    end = clock();
    std::cout << "run time:" << (double) (end - start) / CLOCKS_PER_SEC << "S" << std::endl;
    return 0;
}
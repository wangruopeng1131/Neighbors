//
// Created by wangr on 2023/10/3.
//

#ifndef NEIGHBORS_UTLIS_H
#define NEIGHBORS_UTLIS_H

#include <iostream>
#include <fstream>
#include "Eigen/Dense"
// 保存Eigen::MatrixXd矩阵为txt
void  saveMatrix2txt(Eigen::MatrixXd &mat, const char *filename)
{
    std::ofstream outfile(filename);
    if (!outfile.is_open())
    {
        std::cout << "open file error" << std::endl;
        return;
    }
    for (int i = 0; i < mat.rows(); i++)
    {
        for (int j = 0; j < mat.cols(); j++)
        {
            outfile << mat(i, j) << " ";
        }
        outfile << std::endl;
    }
    outfile.close();
}

#endif //NEIGHBORS_UTLIS_H

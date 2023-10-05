#include <iostream>
#include <ctime>
#include <fstream>

#include "KDTree.h"
#include "BallTree.h"
#include "utlis.h"

int main()
{
    Eigen::setNbThreads(8);
    std::ifstream d(R"(D:\project\Neighbors\test_data)");
    Eigen::MatrixXd data = Eigen::MatrixXd::Zero(100000, 10);
    std::string lines;
    std::size_t pos{};
    int i = 0;
    int j = 0;
    while (std::getline(d, lines))
    {
//        lines.pop_back();
        double temp = std::stod(lines, &pos);
        data(j, i) = temp;
        i += 1;
        if (i >= 10)
        {
            i = 0;
            j += 1;
        }
    }
    d.close();
    clock_t start, end;

    KDTree KdTree = KDTree(data, 40);

    start = clock();
    auto [dualBreadth, idx1] = KdTree.query(data, 6, true, true);
    cout << "KdTree dualBreadth run time:" << (double) (clock() - start) / CLOCKS_PER_SEC << "S" << endl;
    saveMatrix2txt(dualBreadth, R"(D:\project\Neighbors\dualBreadth)");

    start = clock();
    auto [dualDepth, idx2] = KdTree.query(data, 6, true, false);
    cout << "KdTree dualDepth run time:" << (double) (clock() - start) / CLOCKS_PER_SEC << "S" << endl;
    saveMatrix2txt(dualDepth, R"(D:\project\Neighbors\dualDepth)");

    start = clock();
    auto [singleBreadth, idx3] = KdTree.query(data, 6, false, true);
    cout << "KdTree singleBreadth run time:" << (double) (clock() - start) / CLOCKS_PER_SEC << "S" << endl;
    saveMatrix2txt(singleBreadth, R"(D:\project\Neighbors\singleBreadth)");

    start = clock();
    auto [singleDepth, idx4] = KdTree.query(data, 6, false, false);
    cout << "KdTree singleDepth run time:" << (double) (clock() - start) / CLOCKS_PER_SEC << "S" << endl;
    saveMatrix2txt(singleDepth, R"(D:\project\Neighbors\singleDepth)");

    BallTree ballTree = BallTree(data, 40);
    start = clock();
    auto [ballDualBreadth, idx5] = ballTree.query(data, 6, true, true);
    cout << "BallTree dualBreadth run time:" << (double) (clock() - start) / CLOCKS_PER_SEC << "S" << endl;
    saveMatrix2txt(ballDualBreadth, R"(D:\project\Neighbors\BallTree dualBreadth)");

    start = clock();
    auto [ballDualDepth, idx6] = ballTree.query(data, 6, true, false);
    cout << "BallTree dualDepth run time:" << (double) (clock() - start) / CLOCKS_PER_SEC << "S" << endl;
    saveMatrix2txt(ballDualDepth, R"(D:\project\Neighbors\BallTree dualDepth)");

    start = clock();
    auto [ballSingleBreadth, idx7] = ballTree.query(data, 6, false, true);
    cout << "BallTree singleBreadth run time:" << (double) (clock() - start) / CLOCKS_PER_SEC << "S" << endl;
    saveMatrix2txt(ballSingleBreadth, R"(D:\project\Neighbors\BallTree singleBreadth)");

    start = clock();
    auto [ballSingleDepth, idx8] = ballTree.query(data, 6, false, false);
    cout << "BallTree singleDepth run time:" << (double) (clock() - start) / CLOCKS_PER_SEC << "S" << endl;
    saveMatrix2txt(ballSingleDepth, R"(D:\project\Neighbors\BallTree singleDepth)");
    return 0;
}

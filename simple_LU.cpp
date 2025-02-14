#include <mpi.h>
#include <vector>
#include <iostream>
using namespace std;

// 定义一个结构体用于存储稀疏矩阵的一行（仅保存非零元素）  
struct SparseRow {
    vector<int> cols;     // 存储非零元素对应的列索引
    vector<double> vals;  // 存储对应的非零值
};

int main(int argc, char** argv) {
    // 初始化MPI环境
    MPI_Init(&argc, &argv);

    int rank;
    int size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // 当前进程编号
    MPI_Comm_size(MPI_COMM_WORLD, &size); // 总进程数

    // 定义问题规模，本例为5行
    int n(5);

    // 定义稀疏矩阵A（每一行用SparseRow存储），向量b和解向量x（初始化为0）
    vector<SparseRow> A(n);
    vector<double> b(n, 0.0);
    vector<double> x(n, 0.0);

    // 定义每一行的层次（level scheduling），同一层内的行可并行计算
    // 在本例中，我们预先手动给出每行的层次：
    // 行0和行1独立，不依赖其他行，故设为level 0；
    // 行2依赖于行1，设为level 1；
    // 行3依赖于行0，设为level 1；
    // 行4依赖于行2和行3，设为level 2。
    vector<int> levels(n, 0);
    levels[0] = 0;
    levels[1] = 0;
    levels[2] = 1;
    levels[3] = 1;
    levels[4] = 2;

    // --- 定义稀疏下三角矩阵 A 及向量 b ---  
    // 为了简单起见，这里在所有进程中都构造相同的矩阵和向量数据

    // 行0：仅包含对角元 A[0][0] = 2.0
    A[0].cols.push_back(0);
    A[0].vals.push_back(2.0);

    // 行1：仅包含对角元 A[1][1] = 3.0
    A[1].cols.push_back(1);
    A[1].vals.push_back(3.0);

    // 行2：包含 A[2][1] = 1.0 和对角元 A[2][2] = 4.0  
    // 计算 x[2] 时需要用到 x[1]（已在level 0计算）
    A[2].cols.push_back(1);
    A[2].vals.push_back(1.0);
    A[2].cols.push_back(2);
    A[2].vals.push_back(4.0);

    // 行3：包含 A[3][0] = 1.0 和对角元 A[3][3] = 5.0  
    // 计算 x[3] 时需要用到 x[0]（已在level 0计算）
    A[3].cols.push_back(0);
    A[3].vals.push_back(1.0);
    A[3].cols.push_back(3);
    A[3].vals.push_back(5.0);

    // 行4：包含 A[4][2] = 2.0, A[4][3] = 3.0 和对角元 A[4][4] = 6.0  
    // 计算 x[4] 时需要用到 x[2]和 x[3]（已在level 1计算）
    A[4].cols.push_back(2);
    A[4].vals.push_back(2.0);
    A[4].cols.push_back(3);
    A[4].vals.push_back(3.0);
    A[4].cols.push_back(4);
    A[4].vals.push_back(6.0);

    // 定义右侧向量 b（可以自行更改）
    b[0] = 2.0;
    b[1] = 9.0;
    b[2] = 8.0;
    b[3] = 10.0;
    b[4] = 12.0;

    // 计算问题中最大的层数
    int maxLevel = 0;
    for (int i = 0; i < n; ++i) {
        if (levels[i] > maxLevel)
            maxLevel = levels[i];
    }
    
    // --- 开始按照层次顺序求解 ---  
    // 对于每一层，所有属于该层的行均可并行计算，
    // 这里采用简单的分配：如果 (row_index % size == rank) 则该进程负责计算该行
    for (int level = 0; level <= maxLevel; level++) {
        // 遍历所有行，找到属于当前层的行
        for (int i = 0; i < n; i++) {
            if (levels[i] == level) {
                // 判断当前行是否由本进程负责（简单的轮流分配）
                if ((i % size) == rank) {
                    double sum = 0.0;  // 用于累加非对角项的乘积
                    double diag = 0.0; // 对角元

                    // 遍历行 i 的所有非零元素
                    for (size_t k = 0; k < A[i].cols.size(); k++) {
                        int col = A[i].cols[k];
                        double val = A[i].vals[k];
                        if (col == i) {
                            // 找到对角元
                            diag = val;
                        } 
                        else {
                            // 累加非对角项，注意此处依赖的 x[col] 应在之前的层次已计算好
                            sum += val * x[col];
                        }
                    }
                    // 使用前向替换公式计算 x[i]
                    // x[i] = (b[i] - sum) / A[i][i]
                    x[i] = (b[i] - sum) / diag;

                    // 输出调试信息，表明本进程计算了哪一行及其结果
                    cout << "Rank " << rank << " computed x[" << i << "] = " << x[i]
                         << " at level " << level << endl;
                }
            }
        }

        // --- 同步各进程计算的结果 ---
        // 当前层所有行计算完后，各进程必须获得其他进程计算的 x 的最新值，
        // 以便后续层次中使用这些结果。
        // 这里采用 MPI_Allreduce 将所有进程的 x 向量“汇总”。
        // 由于每个行仅由一个进程负责计算，其它进程对应该行的 x[i] 仍为 0，
        // 使用求和操作即可得到正确的 x 向量。
        vector<double> x_global(n, 0.0);
        MPI_Allreduce(x.data(), x_global.data(), n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        // 更新本进程的 x 向量
        x = x_global;

        // 使用 MPI_Barrier 确保所有进程都完成了本层次计算与通信
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // 最后，由 rank 0 输出最终求得的解向量 x
    if (rank == 0) {
        cout << "Solution x: ";
        for (int i = 0; i < n; i++) {
            cout << x[i] << " ";
        }
        cout << endl;
    }

    // 结束MPI环境
    MPI_Finalize();
    return 0;
}

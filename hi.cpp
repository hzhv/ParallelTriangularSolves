#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <omp.h>

using namespace std;
using Clock = std::chrono::high_resolution_clock;

struct CSRMatrix {
    int n; // 方阵维度
    vector<double> values;   // 非零元素数组
    vector<int> col_idx;     // 每个非零元素对应的列下标
    vector<int> row_ptr;     // 长度为 n+1，第 i 行的起始位置在 values 中的索引
};

// 在 CSR 矩阵中查找第 row 行中是否存在目标列 target，若存在返回对应在 values 中的索引，否则返回 -1
int findPosition(const CSRMatrix &A, int row, int target) {
    for (int idx = A.row_ptr[row]; idx < A.row_ptr[row + 1]; idx++) {
        if (A.col_idx[idx] == target)
            return idx;
    }
    return -1;
}
 
// 从 CSR 矩阵中获取第 row 行第 col 列的值（若不存在则返回 0）
double getValue(const CSRMatrix &A, int row, int col) {
    int pos = findPosition(A, row, col);
    return (pos != -1) ? A.values[pos] : 0.0;
}
 
// 生成一个 n x n 的稀疏矩阵，每行大约 nnz_per_row 个非零元素（确保对角元非零）
CSRMatrix generateSparseMatrix(int n, int nnz_per_row) {
    CSRMatrix A;
    A.n = n;
    A.row_ptr.resize(n + 1, 0);
    
    // 先计算 row_ptr，每行至少包含对角元
    for (int i = 0; i < n; i++) {
        // 如果 nnz_per_row 小于 1，则至少保证对角元
        int nnz = max(1, nnz_per_row);
        A.row_ptr[i + 1] = A.row_ptr[i] + nnz;
    }
    int total_nnz = A.row_ptr[n];
    A.values.resize(total_nnz);
    A.col_idx.resize(total_nnz);
    
    // 对每一行生成非零列索引
    for (int i = 0; i < n; i++) {
        vector<int> cols;
        // 保证对角元存在
        cols.push_back(i);
        // 随机生成其他 nnz_per_row-1 个不重复的列下标
        while (cols.size() < (size_t)max(1, nnz_per_row)) {
            int col = rand() % n;
            if (find(cols.begin(), cols.end(), col) == cols.end())
                cols.push_back(col);
        }
        // 排序便于后续查找（保证每行的列下标从小到大）
        sort(cols.begin(), cols.end());
        // 填充 CSR 结构中该行的非零列和数值（数值设为 1~10 的随机浮点数，若对角元，确保非零）
        int start = A.row_ptr[i];
        for (size_t j = 0; j < cols.size(); j++) {
            A.col_idx[start + j] = cols[j];
            // 对角元赋予较大值以保证分解过程稳定
            if (cols[j] == i)
                A.values[start + j] = 10.0 + (rand() % 10) / 10.0;
            else
                A.values[start + j] = 1.0 + (rand() % 10) / 10.0;
        }
    }
    
    return A;
}
 
// 将 CSR 矩阵转换为密集矩阵（二维 vector），仅用于全 LU 分解的演示
vector<vector<double>> convertToDense(const CSRMatrix &A) {
    int n = A.n;
    vector<vector<double>> dense(n, vector<double>(n, 0.0));
    for (int i = 0; i < n; i++) {
        for (int idx = A.row_ptr[i]; idx < A.row_ptr[i + 1]; idx++) {
            int col = A.col_idx[idx];
            dense[i][col] = A.values[idx];
        }
    }
    return dense;
}
 
// 深拷贝一个 CSRMatrix
CSRMatrix copyCSR(const CSRMatrix &A) {
    CSRMatrix B;
    B.n = A.n;
    B.values = A.values;
    B.col_idx = A.col_idx;
    B.row_ptr = A.row_ptr;
    return B;
}
 
// ========== 全 LU 分解（密集矩阵版 Doolittle 算法） ==========
// 注意：此处仅作为演示，不针对稀疏性进行优化
void fullLU(const vector<vector<double>> &A_input) {
    int n = A_input.size();
    // 创建 A 的拷贝用于原地分解
    vector<vector<double>> A = A_input;
    vector<vector<double>> L(n, vector<double>(n, 0.0));
    vector<vector<double>> U(n, vector<double>(n, 0.0));
 
    for (int i = 0; i < n; i++) {
        // 计算 U 的第 i 行（i <= j < n）
        for (int j = i; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < i; k++) {
                sum += L[i][k] * U[k][j];
            }
            U[i][j] = A[i][j] - sum;
        }
        // 计算 L 的第 i 列（i+1 <= j < n）
        L[i][i] = 1.0; // 单位对角线
        for (int j = i + 1; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < i; k++) {
                sum += L[j][k] * U[k][i];
            }
            L[j][i] = (A[j][i] - sum) / U[i][i];
        }
    }
    // 可选：输出 LU 分解结果
    // (这里不输出，以免数据量太大)
}
 
// ========== ILU(0) 分解（串行版） ==========
// 算法思路：按行依次处理，
// 对于行 i，处理所有非零项 A(i,j)（j < i）：
//    A(i,j) = A(i,j) / A(j,j)
//    然后对每个 j < i，对 row j 中 j< k 的非零元素，如果 row i 也有 k，则更新：
//      A(i,k) = A(i,k) - A(i,j) * A(j,k)
// 其中更新仅在原有非零结构内进行（ILU(0)）
void ILU0_serial(CSRMatrix &A) {
    int n = A.n;
    // 依次对每一行进行分解
    for (int i = 0; i < n; i++) {
        // 遍历 row i 的所有非零元素
        for (int idx = A.row_ptr[i]; idx < A.row_ptr[i + 1]; idx++) {
            int j = A.col_idx[idx];
            if (j < i) {  // 仅处理下三角部分
                // 获取对角元 A(j,j)
                double diag = getValue(A, j, j);
                A.values[idx] /= diag; // 存储 L(i,j)
                double multiplier = A.values[idx];
                // 使用 row j 更新 row i
                for (int idx_j = A.row_ptr[j]; idx_j < A.row_ptr[j + 1]; idx_j++) {
                    int col_j = A.col_idx[idx_j];
                    if (col_j > j) {
                        // 若 row i 存在该列，则更新
                        int pos = findPosition(A, i, col_j);
                        if (pos != -1) {
                            A.values[pos] -= multiplier * A.values[idx_j];
                        }
                    }
                }
            }
        }
    }
}
 
// ========== 计算 ILU 分解的层次（Level Scheduling） ==========
// 对于每一行 i，令 level[i] = 0，
// 然后对 row i 中每个 j (< i) 存在非零，令 level[i] = max(level[i], level[j] + 1)
vector<int> computeLevels(const CSRMatrix &A, int &max_level) {
    int n = A.n;
    vector<int> level(n, 0);
    max_level = 0;
    for (int i = 0; i < n; i++) {
        int lev = 0;
        // 遍历 row i 的下三角部分（j < i）
        for (int idx = A.row_ptr[i]; idx < A.row_ptr[i + 1]; idx++) {
            int j = A.col_idx[idx];
            if (j < i) {
                lev = max(lev, level[j] + 1);
            }
        }
        level[i] = lev;
        max_level = max(max_level, lev);
    }
    return level;
}
 
// ========== ILU(0) 分解（并行版，基于层次调度） ==========
// 按照层次依次处理，每一层内的行可并行更新
void ILU0_parallel(CSRMatrix &A, const vector<int> &level, int max_level) {
    int n = A.n;
    // 对每一层依次处理
    for (int lev = 0; lev <= max_level; lev++) {
        // 收集当前层的所有行
        vector<int> rows;
        for (int i = 0; i < n; i++) {
            if (level[i] == lev)
                rows.push_back(i);
        }
        // 并行处理当前层的各行（使用 OpenMP）
        #pragma omp parallel for schedule(dynamic)
        for (size_t idx = 0; idx < rows.size(); idx++) {
            int i = rows[idx];
            // 与串行版相同的 ILU(0) 处理
            for (int pos = A.row_ptr[i]; pos < A.row_ptr[i + 1]; pos++) {
                int j = A.col_idx[pos];
                if (j < i) {
                    double diag = getValue(A, j, j);
                    A.values[pos] /= diag;
                    double multiplier = A.values[pos];
                    for (int pos_j = A.row_ptr[j]; pos_j < A.row_ptr[j + 1]; pos_j++) {
                        int col_j = A.col_idx[pos_j];
                        if (col_j > j) {
                            int pos2 = findPosition(A, i, col_j);
                            if (pos2 != -1) {
                                A.values[pos2] -= multiplier * A.values[pos_j];
                            }
                        }
                    }
                }
            }
        } // end parallel for
    }
}


int main() {
    srand((unsigned)time(0));
    
    int n = 1000;          
    int nnz_per_row = 10;      // 每行非零数（至少 1 个对角元）
    
    cout << "生成 " << n << "x" << n << " 的稀疏矩阵，每行约 " << nnz_per_row << " 个非零元素..." << endl;
    CSRMatrix A_csr = generateSparseMatrix(n, nnz_per_row);

    // 1. 全 LU 分解（先转换为密集矩阵）
    cout << "转换为密集矩阵，并进行全 LU 分解..." << endl;
    vector<vector<double>> A_dense = convertToDense(A_csr);
    
    auto start = Clock::now();
    fullLU(A_dense);
    auto end = Clock::now();
    double duration_fullLU = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    cout << "全 LU 分解耗时: " << duration_fullLU << " 毫秒" << endl;

    // 2. 串行 ILU(0) 分解
    cout << "执行串行 ILU(0) 分解..." << endl;
    // 为保证每个方法使用相同的初始矩阵，深拷贝 CSR 矩阵
    CSRMatrix A_ilu_serial = copyCSR(A_csr);
    // 重复执行 ILU0_serial 多次计时
    int iterations = 1000;
    start = Clock::now();
    for (int iter = 0; iter < iterations; iter++) {
        CSRMatrix A_temp = copyCSR(A_csr);
        ILU0_serial(A_temp);
    }
    end = Clock::now();
    double duration_serial_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    cout << "串行 ILU(0) 分解平均耗时: " 
        << duration_serial_us / iterations << " 微秒" << endl;

    // 3. 并行 ILU(0) 分解（基于层次调度）
    cout << "计算层次调度，并执行并行 ILU(0) 分解..." << endl;
    CSRMatrix A_ilu_parallel = copyCSR(A_csr);
    int max_level;
    vector<int> levels = computeLevels(A_ilu_parallel, max_level);
    
    start = Clock::now();
    for (int i = 0; i < iterations; i++) {
        CSRMatrix A_temp = copyCSR(A_csr);
        ILU0_parallel(A_ilu_parallel, levels, max_level);
    }
    end = Clock::now();
    double duration_ilu_parallel = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    cout << "并行 ILU(0) 分解耗时: " 
        << duration_ilu_parallel / iterations << " 微秒" << endl;
    
    return 0;
}

// g++ -O2 -fopenmp -std=c++11 -o ilu_example ilu_example.cpp

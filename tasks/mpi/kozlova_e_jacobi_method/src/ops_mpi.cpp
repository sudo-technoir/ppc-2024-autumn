// Copyright 2023 Nesterov Alexander
#include "mpi/kozlova_e_jacobi_method/include/ops_mpi.hpp"

#include <cmath>
#include <string>
#include <vector>

bool kozlova_e_jacobi_method_mpi::MethodJacobiSeq::pre_processing() {
  internal_order_test();

  auto* rhs = reinterpret_cast<double*>(taskData->inputs[1]);
  auto* initial_guess = reinterpret_cast<double*>(taskData->inputs[2]);
  B.resize(N);
  X.resize(N);

  for (int i = 0; i < N; i++) {
    B[i] = rhs[i];
    X[i] = initial_guess[i];
  }

  return true;
}

int kozlova_e_jacobi_method_mpi::rankOfMatrix(std::vector<double>& matrix, int n) {
  int rank = n;
  std::vector<std::vector<double>> temp(n, std::vector<double>(n));

  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) temp[i][j] = matrix[i * n + j];

  for (int row = 0; row < rank; ++row) {
    if (temp[row][row] == 0) {
      bool swapDone = false;
      for (int i = row + 1; i < n; ++i) {
        if (temp[i][row] != 0) {
          swap(temp[row], temp[i]);
          swapDone = true;
          break;
        }
      }
      if (!swapDone) {
        rank--;
        continue;
      }
    }

    for (int i = row + 1; i < n; ++i) {
      double factor = temp[i][row] / temp[row][row];
      for (int j = row; j < n; ++j) {
        temp[i][j] -= temp[row][j] * factor;
      }
    }
  }

  return rank;
}

bool kozlova_e_jacobi_method_mpi::hasUniqueSolution(std::vector<double>& A, std::vector<double>& b, int n) {
  std::vector<double> extended_matrix = A;
  extended_matrix.insert(extended_matrix.end(), b.begin(), b.end());

  int rankA = rankOfMatrix(A, n);

  int extended_rank = rankOfMatrix(extended_matrix, n);

  return rankA == extended_rank && rankA == n;
}

bool kozlova_e_jacobi_method_mpi::MethodJacobiSeq::validation() {
  internal_order_test();

  auto* matrix = reinterpret_cast<double*>(taskData->inputs[0]);
  N = static_cast<int>(taskData->inputs_count[0]);
  auto* rhs = reinterpret_cast<double*>(taskData->inputs[1]);
  A.resize(N * N);
  B.resize(N);
  eps = *reinterpret_cast<double*>(taskData->inputs[3]);
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      A[i * N + j] = matrix[i * N + j];
    }
    B[i] = rhs[i];
  }

  for (int i = 0; i < N; i++) {
    if (A[i * N + i] == 0) {
      std::cerr << "Incorrect matrix: diagonal element A[" << i + 1 << "][" << i + 1 << "] is zero." << std::endl;
      return false;
    }
  }
  if (!hasUniqueSolution(A, B, N)) {
    std::cerr << "The matrix may not have a single solution" << std::endl;
    return false;
  }
  if (eps <= 0.0) {
    std::cerr << "Epsilon less zero!" << std::endl;
    return false;
  }
  return taskData->inputs_count[0] > 0;
}

void kozlova_e_jacobi_method_mpi::MethodJacobiSeq::jacobi_iteration() {
  std::vector<double> TempX(N);

  for (int i = 0; i < N; i++) {
    TempX[i] = B[i];
    for (int j = 0; j < N; j++) {
      if (i != j) TempX[i] -= A[i * N + j] * X[j];
    }
    TempX[i] /= A[i * N + i];
  }

  for (int h = 0; h < N; h++) {
    X[h] = TempX[h];
  }
}

bool kozlova_e_jacobi_method_mpi::MethodJacobiSeq::run() {
  internal_order_test();
  double norm;
  std::vector<double> prev_X(N);
  do {
    prev_X = X;

    jacobi_iteration();

    norm = fabs(X[0] - prev_X[0]);
    for (int i = 0; i < N; i++) {
      if (fabs(X[i] - prev_X[i]) > norm) norm = fabs(X[i] - prev_X[i]);
    }
  } while (norm > eps);
  return true;
}

bool kozlova_e_jacobi_method_mpi::MethodJacobiSeq::post_processing() {
  internal_order_test();
  for (int i = 0; i < N; i++) {
    reinterpret_cast<double*>(taskData->outputs[0])[i] = X[i];
  }
  return true;
}

bool kozlova_e_jacobi_method_mpi::MethodJacobiMPI::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    auto* rhs = reinterpret_cast<double*>(taskData->inputs[1]);
    auto* initial_guess = reinterpret_cast<double*>(taskData->inputs[2]);
    B.resize(N);
    X.resize(N);

    for (int i = 0; i < N; i++) {
      B[i] = rhs[i];
      X[i] = initial_guess[i];
    }
  }
  return true;
}

void kozlova_e_jacobi_method_mpi::MethodJacobiMPI::jacobi_iteration() {
  std::vector<double> TempX(N, 0.0);
  int rank = world.rank();
  int size = world.size();

  int rows_per_process = (N + size - 1) / size;
  int start_row = rank * rows_per_process;
  int end_row = std::min((rank + 1) * rows_per_process, N);

  for (int i = start_row; i < end_row; i++) {
    TempX[i] = B[i];
    for (int j = 0; j < N; j++) {
      if (i != j) {
        TempX[i] -= A[i * N + j] * X[j];
      }
    }
    TempX[i] /= A[i * N + i];
  }

  std::vector<int> sendcounts(size, rows_per_process);
  if (size > N) {
    for (size_t i = N; i < sendcounts.size(); i++) sendcounts[i] = 0;
  }
  std::vector<int> displacements(size, 0);

  sendcounts[size - 1] = (N - (size - 1) * rows_per_process) < 0 ? 0 : N - (size - 1) * rows_per_process;
  for (int i = 1; i < size; i++) {
    displacements[i] = displacements[i - 1] + sendcounts[i - 1];
  }
  int extra_send = 0;
  for (size_t i = 0; i < sendcounts.size(); i++) extra_send += sendcounts[i];
  std::vector<double> all_X(N + extra_send, 0.0);

  boost::mpi::gatherv(world, TempX.data() + start_row, sendcounts[rank], all_X.data(), sendcounts, displacements, 0);

  if (rank == 0) {
    all_X.resize(N);
    X = all_X;
  }

  boost::mpi::broadcast(world, X.data(), X.size(), 0);
}

bool kozlova_e_jacobi_method_mpi::MethodJacobiMPI::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    auto* matrix = reinterpret_cast<double*>(taskData->inputs[0]);
    N = static_cast<int>(taskData->inputs_count[0]);
    eps = *reinterpret_cast<double*>(taskData->inputs[3]);
    auto* rhs = reinterpret_cast<double*>(taskData->inputs[1]);
    A.resize(N * N);
    B.resize(N);
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        A[i * N + j] = matrix[i * N + j];
      }
      B[i] = rhs[i];
    }
    for (int i = 0; i < N; i++) {
      if (A[i * N + i] == 0) {
        std::cerr << "Incorrect matrix: diagonal element A[" << i + 1 << "][" << i + 1 << "] is zero." << std::endl;
        return false;
      }
    }
    if (!hasUniqueSolution(A, B, N)) {
      std::cerr << "The matrix may not have a single solution" << std::endl;
      return false;
    }
    if (eps <= 0.0) {
      std::cerr << "Epsilon less zero!" << std::endl;
      return false;
    }
    return taskData->inputs_count[0] > 0;
  }
  return true;
}

bool kozlova_e_jacobi_method_mpi::MethodJacobiMPI::run() {
  internal_order_test();

  boost::mpi::broadcast(world, N, 0);
  A.resize(N * N);
  B.resize(N);
  X.resize(N);
  boost::mpi::broadcast(world, A.data(), A.size(), 0);
  boost::mpi::broadcast(world, B.data(), B.size(), 0);
  boost::mpi::broadcast(world, X.data(), X.size(), 0);
  boost::mpi::broadcast(world, eps, 0);

  double norm;
  std::vector<double> prev_X(N);
  do {
    prev_X = X;

    jacobi_iteration();

    norm = 0;
    for (int i = 0; i < N; i++) {
      norm = std::max(norm, fabs(X[i] - prev_X[i]));
    }
  } while (norm > eps);

  return true;
}

bool kozlova_e_jacobi_method_mpi::MethodJacobiMPI::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    for (int i = 0; i < N; i++) {
      reinterpret_cast<double*>(taskData->outputs[0])[i] = X[i];
    }
  }
  return true;
}
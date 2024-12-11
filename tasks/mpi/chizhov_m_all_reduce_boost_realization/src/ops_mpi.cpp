// Copyright 2023 Nesterov Alexander
#include "mpi/chizhov_m_all_reduce_boost_realization/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <vector>

bool chizhov_m_all_reduce_boost_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  // Init vectors

  cols = taskData->inputs_count[1];
  rows = taskData->inputs_count[2];

  auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
  input_.assign(tmp_ptr, tmp_ptr + taskData->inputs_count[0]);

  res_ = std::vector<int>(cols, INT_MIN);
  sum = std::vector<int>(cols, 0);

  return true;
}

bool chizhov_m_all_reduce_boost_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  return (taskData->inputs_count[0] > 0 && taskData->inputs_count[2] != 0 && !taskData->inputs.empty() &&
          taskData->inputs_count[1] != 0 && (taskData->inputs_count[1] == taskData->outputs_count[0]));
}

bool chizhov_m_all_reduce_boost_mpi::TestMPITaskSequential::run() {
  internal_order_test();

  for (int j = 0; j < cols; j++) {
    int maxElement = input_[j];
    for (int i = 1; i < rows; i++) {
      if (input_[i * cols + j] > maxElement) {
        maxElement = input_[i * cols + j];
      }
    }
    res_[j] = maxElement;
  }

  for (int j = 0; j < cols; j++) {
    for (int i = 0; i < rows; i++) {
      if (input_[i * cols + j] < res_[j]) {
        sum[j]++;
      }
    }
  }
  return true;
}

bool chizhov_m_all_reduce_boost_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  for (int i = 0; i < cols; i++) {
    reinterpret_cast<int*>(taskData->outputs[0])[i] = sum[i];
  }
  return true;
}

bool chizhov_m_all_reduce_boost_mpi::TestMPITaskBoostParallel::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    cols = taskData->inputs_count[1];
    rows = taskData->inputs_count[2];
  }

  if (world.rank() == 0) {
    // init vectors
    auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
    input_.assign(tmp_ptr, tmp_ptr + taskData->inputs_count[0]);
  } else {
    input_ = std::vector<int>(cols * rows, 0);
  }

  return true;
}

bool chizhov_m_all_reduce_boost_mpi::TestMPITaskBoostParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return (taskData->inputs_count[0] > 0 && taskData->inputs_count[2] != 0 && !taskData->inputs.empty() &&
            taskData->inputs_count[1] != 0 && (taskData->inputs_count[1] == taskData->outputs_count[0]));
  }
  return true;
}

bool chizhov_m_all_reduce_boost_mpi::TestMPITaskBoostParallel::run() {
  internal_order_test();

  broadcast(world, cols, 0);
  broadcast(world, rows, 0);

  if (world.rank() != 0) {
    input_ = std::vector<int>(cols * rows, 0);
  }
  broadcast(world, input_.data(), cols * rows, 0);

  int delta = cols / world.size();
  int extra = cols % world.size();
  if (extra != 0) {
    delta += 1;
  }
  int startCol = delta * world.rank();
  int lastCol = std::min(cols, delta * (world.rank() + 1));
  std::vector<int> localMax(cols, INT_MIN);
  for (int j = startCol; j < lastCol; j++) {
    int maxElem = input_[j];
    for (int i = 1; i < rows; i++) {
      int coor = i * cols + j;
      if (input_[coor] > maxElem) {
        maxElem = input_[coor];
      }
    }
    localMax[j] = maxElem;
  }
  res_.resize(cols, INT_MIN);
  boost::mpi::all_reduce(world, localMax.data(), cols, res_.data(), boost::mpi::maximum<int>());

  std::vector<int> local_cnt_(cols, 0);
  for (int j = startCol; j < lastCol; j++) {
    for (int i = 0; i < rows; i++) {
      int coor = i * cols + j;
      if (input_[coor] < res_[j]) {
        local_cnt_[j]++;
      }
    }
  }
  sum.resize(cols, 0);
  boost::mpi::reduce(world, local_cnt_.data(), cols, sum.data(), std::plus<>(), 0);
  return true;
}

bool chizhov_m_all_reduce_boost_mpi::TestMPITaskBoostParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    for (int i = 0; i < cols; i++) {
      reinterpret_cast<int*>(taskData->outputs[0])[i] = sum[i];
    }
  }
  return true;
}
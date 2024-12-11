// Copyright 2024 Nesterov Alexander
#include "seq/titov_s_simple_iteration/include/ops_seq.hpp"

#include <thread>

void titov_s_simple_iteration_seq::SimpleIterationSequential::transformMatrix() {
  for (unsigned int i = 0; i < rows_; ++i) {
    float diagonal = input_[i * cols_ + i];

    for (unsigned int j = 0; j < cols_ - 1; ++j) {
      if (i != j) {
        input_[i * cols_ + j] = -input_[i * cols_ + j] / diagonal;
      } else {
        input_[i * cols_ + j] = 0.0f;
      }
    }

    input_[i * cols_ + cols_ - 1] /= diagonal;
  }
}

bool titov_s_simple_iteration_seq::SimpleIterationSequential::isDiagonallyDominant() {
  for (unsigned int i = 0; i < rows_; ++i) {
    float diagonal = std::abs(input_[i * cols_ + i]);
    float sum = 0.0f;

    for (unsigned int j = 0; j < cols_ - 1; ++j) {
      if (i != j) {
        sum += std::abs(input_[i * cols_ + j]);
      }
    }

    if (diagonal <= sum) {
      return false;
    }
  }

  return true;
}

bool titov_s_simple_iteration_seq::SimpleIterationSequential::validation() {
  internal_order_test();

  if (taskData->inputs_count.empty() || taskData->inputs.empty()) {
    return false;
  }

  unsigned int rows = taskData->inputs_count[0];
  unsigned int cols = taskData->inputs_count[1];

  if (rows < cols - 1) {
    return false;
  }

  auto* epsilon_ptr = reinterpret_cast<float*>(taskData->inputs[rows]);

  float epsilon = *epsilon_ptr;
  if (epsilon <= 0.0f || epsilon > 1.0f) {
    return false;
  }

  if (taskData->outputs_count.empty() || taskData->outputs_count[0] < 1) {
    return false;
  }

  return true;
}

bool titov_s_simple_iteration_seq::SimpleIterationSequential::pre_processing() {
  internal_order_test();

  rows_ = taskData->inputs_count[0];
  cols_ = taskData->inputs_count[1];

  input_ = std::make_unique<float[]>(rows_ * cols_);
  res_ = std::make_unique<float[]>(rows_);

  for (unsigned int i = 0; i < rows_; i++) {
    auto* tmp_ptr = reinterpret_cast<float*>(taskData->inputs[i]);
    for (unsigned int j = 0; j < cols_; j++) {
      input_[i * cols_ + j] = tmp_ptr[j];
    }
  }

  auto* epsilon_ptr = reinterpret_cast<float*>(taskData->inputs[rows_]);
  epsilon_ = *epsilon_ptr;

  if (!isDiagonallyDominant()) {
    return false;
  }

  transformMatrix();

  return true;
}

bool titov_s_simple_iteration_seq::SimpleIterationSequential::run() {
  internal_order_test();

  std::unique_ptr<float[]> x_prev = std::make_unique<float[]>(rows_);
  std::unique_ptr<float[]> x_curr = std::make_unique<float[]>(rows_);
  std::fill(x_prev.get(), x_prev.get() + rows_, 0.0f);

  float max_diff;

  do {
    max_diff = 0.0f;

    for (unsigned int i = 0; i < rows_; i++) {
      float new_x = input_[i * cols_ + cols_ - 1];

      for (unsigned int j = 0; j < cols_ - 1; j++) {
        new_x += input_[i * cols_ + j] * x_prev[j];
      }

      x_curr[i] = new_x;

      float diff = std::abs(x_curr[i] - x_prev[i]);
      max_diff = std::max(max_diff, diff);
    }

    std::copy(x_curr.get(), x_curr.get() + rows_, x_prev.get());

  } while (max_diff > epsilon_);

  std::copy(x_curr.get(), x_curr.get() + rows_, res_.get());

  return true;
}

bool titov_s_simple_iteration_seq::SimpleIterationSequential::post_processing() {
  internal_order_test();

  auto* output_ptr = reinterpret_cast<float*>(taskData->outputs[0]);

  for (unsigned int i = 0; i < rows_; ++i) {
    output_ptr[i] = res_[i];
  }

  return true;
}

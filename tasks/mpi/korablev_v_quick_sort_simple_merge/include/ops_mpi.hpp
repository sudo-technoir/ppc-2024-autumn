#pragma once

#include <gtest/gtest.h>

#include <boost/mpi.hpp>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace korablev_v_quick_sort_simple_merge_mpi {

class QuickSortSimpleMergeSequential : public ppc::core::Task {
 public:
  explicit QuickSortSimpleMergeSequential(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input_;
  std::vector<int> output_;
};

class QuickSortSimpleMergeParallel : public ppc::core::Task {
 public:
  explicit QuickSortSimpleMergeParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input_;
  std::vector<int> output_;
  std::vector<int> local_data_;
  boost::mpi::communicator world;
};

inline std::vector<int> merge(const std::vector<int>& left, const std::vector<int>& right);
inline std::vector<int> quick_sort_with_merge(const std::span<int>& arr);

}  // namespace korablev_v_quick_sort_simple_merge_mpi

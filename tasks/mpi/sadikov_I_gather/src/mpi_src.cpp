#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "mpi/sadikov_I_gather/include/ops_mpi.h"

sadikov_I_gather_mpi::MPITask::MPITask(std::shared_ptr<ppc::core::TaskData> td) : Task(std::move(td)) {}

bool sadikov_I_gather_mpi::MPITask::validation() {
  internal_order_test();
  return taskData->inputs_count[0] == taskData->outputs_count[0];
}

bool sadikov_I_gather_mpi::MPITask::pre_processing() {
  internal_order_test();
  m_rowsCount = static_cast<size_t>(taskData->inputs_count[0]);
  m_columnsCount = static_cast<size_t>(taskData->inputs_count[1]);
  auto *tmpPtr = reinterpret_cast<int *>(taskData->inputs[0]);
  m_matrix.reserve(m_columnsCount * m_rowsCount);
  for (size_t i = 0; i < m_columnsCount * m_rowsCount; ++i) {
    m_matrix.emplace_back(tmpPtr[i]);
  }
  m_sum = std::vector<int>(m_rowsCount);
  return true;
}

bool sadikov_I_gather_mpi::MPITask::run() {
  internal_order_test();
  Accumulate(m_rowsCount);
  return true;
}

bool sadikov_I_gather_mpi::MPITask::post_processing() {
  internal_order_test();
  for (size_t i = 0; i < m_rowsCount; ++i) {
    reinterpret_cast<int *>(taskData->outputs[0])[i] = m_sum[i];
  }
  return true;
}

void sadikov_I_gather_mpi::MPITask::Accumulate(size_t size) {
  for (size_t i = 0; i < size; ++i) {
    m_sum[i] = std::accumulate(m_matrix.begin() + i * m_columnsCount, m_matrix.begin() + (i + 1) * m_columnsCount, 0);
  }
}

sadikov_I_gather_mpi::MPITaskParallel::MPITaskParallel(std::shared_ptr<ppc::core::TaskData> td) : Task(std::move(td)) {}

bool sadikov_I_gather_mpi::MPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return taskData->inputs_count[0] == taskData->outputs_count[0];
  }
  return true;
}

bool sadikov_I_gather_mpi::MPITaskParallel::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    m_rowsCount = static_cast<size_t>(taskData->inputs_count[0]);
    m_columnsCount = static_cast<size_t>(taskData->inputs_count[1]);
    m_delta = m_rowsCount / world.size();
    m_lastColumn = m_rowsCount % world.size();
    m_matrix.reserve(m_columnsCount * m_rowsCount);
    int *tmpPtr = reinterpret_cast<int *>(taskData->inputs[0]);
    for (size_t i = 0; i < m_columnsCount * m_rowsCount; ++i) {
      m_matrix.emplace_back(tmpPtr[i]);
    }
    for (int i = 0; i < world.size(); ++i) {
      m_id.emplace(i, i);
    }
    std::swap(m_id[0], m_id[m_root]);
  }
  return true;
}

bool sadikov_I_gather_mpi::MPITaskParallel::run() {
  internal_order_test();
  broadcast(world, m_rowsCount, 0);
  broadcast(world, m_columnsCount, 0);
  broadcast(world, m_delta, 0);
  broadcast(world, m_lastColumn, 0);
  broadcast(world, m_id, 0);
  world.rank() == 0 ? m_localInput.resize((m_delta + m_lastColumn) * m_columnsCount)
                    : m_localInput.resize(m_delta * m_columnsCount);
  std::vector<int> sizes = CalculateSize(m_delta * m_columnsCount, m_columnsCount);
  std::vector<int> displacements(world.size());
  if (m_rowsCount > 1) {
    for (int i = 0; i < world.size(); ++i) {
      if (i == 0) {
        displacements[i] = 0;
      } else {
        displacements[i] = displacements[i - 1] + sizes[i - 1];
      }
    }
    boost::mpi::scatterv(world, m_matrix.data(), sizes, displacements, m_localInput.data(), sizes[world.rank()], 0);
  } else {
    m_localInput = m_matrix;
  }
  size_t size = m_delta != 0 ? m_localInput.size() / m_columnsCount : 1;
  std::vector<int> intermediate_res;
  if (m_delta != 0) {
    intermediate_res = Accumulate(size);
  }
  if (world.rank() == 0 && m_delta == 0 && !m_matrix.empty()) {
    intermediate_res.emplace_back(std::accumulate(m_matrix.begin(), m_matrix.end(), 0));
  }
  if (world.size() > 1 && m_rowsCount > 1) {
    std::vector<int> m_sizes(world.size(), m_delta);
    m_sizes[m_root] += m_lastColumn;
    sadikov_I_gather_mpi::Gather(world, m_id, intermediate_res, m_sizes, m_sum, m_rowsCount, m_root);
  } else {
    m_sum = std::move(intermediate_res);
  }
  return true;
}

bool sadikov_I_gather_mpi::MPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == m_root) {
    for (size_t i = 0; i < m_rowsCount; ++i) {
      reinterpret_cast<int *>(taskData->outputs[0])[i] = m_sum[i];
    }
  }
  return true;
}

std::vector<int> sadikov_I_gather_mpi::MPITaskParallel::Accumulate(size_t size) {
  std::vector<int> in(size);
  for (size_t i = 0; i < size; ++i) {
    in[i] =
        std::accumulate(m_localInput.begin() + i * m_columnsCount, m_localInput.begin() + (i + 1) * m_columnsCount, 0);
  }
  return in;
}

std::vector<int> sadikov_I_gather_mpi::MPITaskParallel::CalculateSize(int elements, int count) {
  std::vector<int> sizes(world.size(), elements);
  if (m_delta == 0 && !m_matrix.empty()) {
    sizes.front() = 1;
  } else if (m_delta != 0 && !m_matrix.empty()) {
    sizes.front() = (m_delta + m_lastColumn) * count;
  }
  return sizes;
}

void sadikov_I_gather_mpi::MPITaskParallel::SetRoot(int root) { m_root = root; }

void sadikov_I_gather_mpi::Gather(const boost::mpi::communicator &world, std::map<int, int> id,
                                  std::vector<int> &inValues, const std::vector<int> &sizes,
                                  std::vector<int> &outValues, int size, int root) {
  std::vector<int> leftChildData(size);
  std::vector<int> rightChildData(size);
  int leftSender = id[world.rank()] * 2 + 1;
  int rightSender = id[world.rank()] * 2 + 2;
  int dataLocation = DataLocation(world.rank(), sizes[root + 1], sizes[root]);
  if (leftSender < world.size() && id[leftSender] < world.size()) {
    std::copy(inValues.begin(), inValues.end(), leftChildData.begin() + dataLocation);
    world.recv(id[leftSender], 0, leftChildData.data() + DataLocation(id[leftSender], sizes[root + 1], sizes[root]),
               size - DataLocation(id[leftSender], sizes[root + 1], sizes[root]));
  }
  if (rightSender < world.size() && id[rightSender] < world.size()) {
    world.recv(id[rightSender], 0, rightChildData.data() + DataLocation(id[rightSender], sizes[root + 1], sizes[root]),
               size - DataLocation(id[rightSender], sizes[root + 1], sizes[root]));
  }
  if (world.rank() != root) {
    int parent = world.rank() * 2 + 1;
    if (world.rank() != 0) {
      parent = world.rank() % 2 == 0 ? (world.rank() - 1) / 2 : world.rank() / 2;
    }
    if (rightSender < world.size()) {
      for (int i = 0; i < size; ++i) {
        if (leftChildData[i] == 0 && rightChildData[i] != 0) {
          leftChildData[i] = rightChildData[i];
        }
      }
    }
    if (leftSender < world.size()) {
      world.send(id[parent], 0, leftChildData.data() + dataLocation, size - dataLocation);
    } else {
      world.send(id[parent], 0, inValues.data(), inValues.size());
    }
  } else {
    outValues.resize(size);
    for (int i = 0; i < size; ++i) {
      if (leftChildData[i] != 0) {
        outValues[i] = leftChildData[i];
      } else if (leftChildData[i] == 0 && rightChildData[i] != 0) {
        outValues[i] = rightChildData[i];
      }
    }
  }
}

int sadikov_I_gather_mpi::DataLocation(int rank, int delta, int maxDelta) {
  return rank == 0 ? 0 : (rank - 1) * delta + maxDelta;
}

sadikov_I_gather_mpi::ReferenceTask::ReferenceTask(std::shared_ptr<ppc::core::TaskData> td) : Task(std::move(td)) {}

bool sadikov_I_gather_mpi::ReferenceTask::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return taskData->inputs_count[0] == taskData->outputs_count[0];
  }
  return true;
}

bool sadikov_I_gather_mpi::ReferenceTask::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    m_rowsCount = static_cast<size_t>(taskData->inputs_count[0]);
    m_columnsCount = static_cast<size_t>(taskData->inputs_count[1]);
    m_delta = m_rowsCount / world.size();
    m_lastColumn = m_rowsCount % world.size();
    m_matrix.reserve(m_columnsCount * m_rowsCount);
    int *tmpPtr = reinterpret_cast<int *>(taskData->inputs[0]);
    for (size_t i = 0; i < m_columnsCount * m_rowsCount; ++i) {
      m_matrix.emplace_back(tmpPtr[i]);
    }
  }
  return true;
}

bool sadikov_I_gather_mpi::ReferenceTask::run() {
  internal_order_test();
  broadcast(world, m_rowsCount, 0);
  broadcast(world, m_columnsCount, 0);
  broadcast(world, m_delta, 0);
  broadcast(world, m_lastColumn, 0);
  world.rank() == 0 ? m_localInput.resize((m_delta + m_lastColumn) * m_columnsCount)
                    : m_localInput.resize(m_delta * m_columnsCount);
  std::vector<int> sizes = CalculateSize(m_delta * m_columnsCount, m_columnsCount);
  std::vector<int> displacements(world.size());
  if (m_rowsCount > 1) {
    for (int i = 0; i < world.size(); ++i) {
      if (i == 0) {
        displacements[i] = 0;
      } else {
        displacements[i] = displacements[i - 1] + sizes[i - 1];
      }
    }
    boost::mpi::scatterv(world, m_matrix.data(), sizes, displacements, m_localInput.data(), sizes[world.rank()], 0);
  } else {
    m_localInput = m_matrix;
  }
  size_t size = m_delta != 0 ? m_localInput.size() / m_columnsCount : 1;
  std::vector<int> intermediate_res;
  if (m_delta != 0) {
    intermediate_res = Accumulate(size);
  }
  if (world.rank() == 0 && m_delta == 0 && !m_matrix.empty()) {
    intermediate_res.emplace_back(std::accumulate(m_matrix.begin(), m_matrix.end(), 0));
  }
  if (world.rank() == 0) {
    std::vector<int> localRes(m_rowsCount);
    boost::mpi::gatherv(world, intermediate_res, localRes.data(), CalculateSize(m_delta, 1), 0);
    m_sum = localRes;
  } else {
    boost::mpi::gatherv(world, intermediate_res, 0);
  }
  return true;
}

bool sadikov_I_gather_mpi::ReferenceTask::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    for (size_t i = 0; i < m_rowsCount; ++i) {
      reinterpret_cast<int *>(taskData->outputs[0])[i] = m_sum[i];
    }
  }
  return true;
}

std::vector<int> sadikov_I_gather_mpi::ReferenceTask::Accumulate(size_t size) {
  std::vector<int> in(size);
  for (size_t i = 0; i < size; ++i) {
    in[i] =
        std::accumulate(m_localInput.begin() + i * m_columnsCount, m_localInput.begin() + (i + 1) * m_columnsCount, 0);
  }
  return in;
}

std::vector<int> sadikov_I_gather_mpi::ReferenceTask::CalculateSize(int elements, int count) {
  std::vector<int> sizes(world.size(), elements);
  if (m_delta == 0 && !m_matrix.empty()) {
    sizes.front() = 1;
  } else if (m_delta != 0 && !m_matrix.empty()) {
    sizes[0] = (m_delta + m_lastColumn) * count;
  }
  return sizes;
}
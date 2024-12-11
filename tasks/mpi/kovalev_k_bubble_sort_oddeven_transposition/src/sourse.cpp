#include "mpi/kovalev_k_bubble_sort_oddeven_transposition/include/header.hpp"

template <class T>
bool kovalev_k_bubble_sort_oddeven_transposition_mpi::BubbleSortOddEvenTranspositionPar<T>::bubble_sort_mpi() {
  for (size_t i = 0; i < loc_v.size() - 1; i++)
    for (size_t j = 0; j < loc_v.size() - i - 1; j++)
      if (loc_v[j] > loc_v[j + 1]) std::swap(loc_v[j], loc_v[j + 1]);
  return true;
}

template <class T>
bool kovalev_k_bubble_sort_oddeven_transposition_mpi::BubbleSortOddEvenTranspositionPar<T>::divide_and_merge(
    int partner) {
  if (partner >= 0 && partner < world.size()) {
    std::vector<T> tmp;
    std::vector<T> res;
    int working_num = std::max(world.rank(), partner);
    if (world.rank() == working_num) {
      tmp.resize(scratter_length);
      world.recv(partner, 0, tmp);
    } else {
      world.send(partner, 0, loc_v);
    }
    if (world.rank() == working_num) {
      res.clear();
      for (size_t i = 0; i < loc_v.size(); i++) tmp.push_back(loc_v[i]);
      size_t iter1 = 0;
      size_t iter2 = scratter_length;
      while (iter2 < tmp.size() || iter1 < scratter_length) {
        if ((iter1 < scratter_length && iter2 < tmp.size() && tmp[iter1] <= tmp[iter2]) ||
            (iter1 < scratter_length && iter2 == tmp.size())) {
          res.push_back(tmp[iter1]);
          iter1++;
        } else if ((iter1 < scratter_length && iter2 < tmp.size() && tmp[iter1] >= tmp[iter2]) ||
                   (iter1 == scratter_length && iter2 < tmp.size())) {
          res.push_back(tmp[iter2]);
          iter2++;
        }
      }
      memcpy(loc_v.data(), res.data() + scratter_length, loc_v.size() * sizeof(T));
    }
    if (world.rank() == working_num) {
      world.send(partner, 0, res.data(), scratter_length);
    } else {
      world.recv(partner, 0, loc_v.data(), scratter_length);
    }
  }
  return true;
}

template <class T>
bool kovalev_k_bubble_sort_oddeven_transposition_mpi::BubbleSortOddEvenTranspositionPar<T>::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    glob_v.resize(n);
    void* ptr_vec = glob_v.data();
    void* ptr_input = taskData->inputs[0];
    memcpy(ptr_vec, ptr_input, sizeof(T) * n);
  }
  return true;
}

template <class T>
bool kovalev_k_bubble_sort_oddeven_transposition_mpi::BubbleSortOddEvenTranspositionPar<T>::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    if (taskData->inputs.empty() || taskData->outputs.empty() || taskData->inputs_count[0] <= 0 ||
        taskData->outputs_count[0] != n || taskData->inputs_count[0] != taskData->outputs_count[0]) {
      return false;
    }
  }
  return true;
}

template <class T>
bool kovalev_k_bubble_sort_oddeven_transposition_mpi::BubbleSortOddEvenTranspositionPar<T>::run() {
  internal_order_test();
  boost::mpi::broadcast(world, n, 0);
  scratter_length = n / world.size();
  if (world.rank() == 0) {
    remainder.resize(n - scratter_length * world.size());
    memcpy(remainder.data(), glob_v.data() + scratter_length * world.size(), sizeof(T) * remainder.size());
  }
  loc_v.resize(scratter_length);
  boost::mpi::scatter(world, glob_v.data(), loc_v.data(), scratter_length, 0);
  bubble_sort_mpi();
  int partner;
  for (int phase = 1; phase <= world.size(); phase++) {
    if (phase % 2 == 1) {
      partner = world.rank() % 2 == 1 ? world.rank() - 1 : world.rank() + 1;
      divide_and_merge(partner);
    } else {
      partner = world.rank() % 2 == 1 ? world.rank() + 1 : world.rank() - 1;
      divide_and_merge(partner);
    }
  }

  if (world.rank() == 0) glob_v.resize(scratter_length * world.size());

  gather(world, loc_v.data(), loc_v.size(), glob_v.data(), 0);

  if (world.rank() == 0) {
    for (size_t i = 0; i < remainder.size(); i++) {
      size_t j = 0;
      while (j < glob_v.size() && remainder[i] > glob_v[j]) j++;
      glob_v.insert(glob_v.begin() + j, remainder[i]);
    }
    remainder.clear();
  }

  return true;
}

template <class T>
bool kovalev_k_bubble_sort_oddeven_transposition_mpi::BubbleSortOddEvenTranspositionPar<T>::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    memcpy(reinterpret_cast<T*>(taskData->outputs[0]), glob_v.data(), sizeof(T) * glob_v.size());
  }
  return true;
}

template class kovalev_k_bubble_sort_oddeven_transposition_mpi::BubbleSortOddEvenTranspositionPar<int>;
template class kovalev_k_bubble_sort_oddeven_transposition_mpi::BubbleSortOddEvenTranspositionPar<double>;
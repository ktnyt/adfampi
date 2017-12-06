#include <iostream>
#include <queue>
#include "mpi.h"
#include <unistd.h>

#include "mnist.hpp"
#include "random.hpp"
#include "activations.hpp"
#include "utils.hpp"

#include "Eigen/Core"

using namespace Eigen;

int main(int argc, char* argv[]) {
  int rank, size;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int root = 0;
  int last = size - 1;
  int epochs = 10;
  int batchsize = 100;

  float lr = 0.05;
  float decay = 0.9995;

  MNIST<float> mnist("mnist");
  mnist.train_images.scale(1.0 / 255.0);
  mnist.test_images.scale(1.0 / 255.0);

  int n_input = mnist.train_images.dims;
  int n_hidden = 240;
  int n_output = 10;

  int n_train = mnist.train_images.length;
  int* perm = new int[n_train];

  for(int i = 0; i < n_train; ++i) {
    perm[i] = i;
  }

  int rows = rank == root ? n_input : n_hidden;
  int cols = rank == last ? n_hidden : n_output;
  int cats = n_output;

  int input_count = batchsize * rows;
  int output_count = batchsize * cols;
  int error_count = batchsize * n_output;
  int label_count = batchsize * n_output;

  float* input;
  float* output;
  float* error;
  float* label;

  int n_steps = epochs * (n_train / batchsize);
  n_steps = 3;

  MPI_Status status;

  int i = 0;

  std::queue<float*> input_queue;
  std::queue<float*> output_queue;
  std::queue<float*> label_queue;

  for(int step = 0; step < n_steps + last; ++step) {
    MPI_Barrier(MPI_COMM_WORLD);

    if(step < n_steps) {
      label = new float[label_count];
  
      if(rank == root) {
        MPI_Send(label, label_count, MPI_FLOAT, last, 0, MPI_COMM_WORLD);
        std::cout << "send label" << std::endl;
      }

      if(rank == last) {
        MPI_Recv(label, label_count, MPI_FLOAT, root, 0, MPI_COMM_WORLD, &status);
        std::cout << "recv label" << std::endl;
      }

      delete[] label;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if(rank < step + 1 && step - rank < n_steps) {
      if(rank != root) {
        input = new float[input_count];
        MPI_Recv(input, input_count, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, &status);
        delete[] input;
        std::cout << rank << ": recv" << std::endl;
      }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if(rank < step + 1 && step - rank < n_steps) {
      if(rank != last) {
        output = new float[output_count];
        MPI_Send(output, output_count, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD);
        delete[] output;
        std::cout << rank << ": send" << std::endl;
      }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if(step + 1 > last) {
      error = new float[error_count];
      MPI_Bcast(error, error_count, MPI_FLOAT, last, MPI_COMM_WORLD);
      delete[] error;
      std::cout << rank << ": bcast" << std::endl;
    }

    i += batchsize;
  }

  delete[] perm;
  
  MPI_Finalize();

  return 0;
}

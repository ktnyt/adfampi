#include <iostream>
#include "mpi.h"
#include "cifar_loader.hpp"
#include "hidden_layer.hpp"
#include "output_layer.hpp"

int main(int argc, char* argv[]) {
  int rank, size;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int root = 0;
  int last = size - 1;
  int epochs = 20;
  int batchsize = 100;

  int n_hidden = 1000;
  int n_output = 10;

  float lr = 0.01;
  float decay = 0.9995;

  if(rank == root) {
    invoke_cifar_loader(rank, last, epochs, batchsize);
  } else if(rank == last) {
    invoke_output_layer(rank, root, n_output, lr);
  } else {
    invoke_hidden_layer(rank, root, last, n_hidden, lr, decay, true);
  }
  
  MPI_Finalize();

  return 0;
}

#include "mnist_loader.hpp"

#include "mpi.h"
#include "Eigen/Core"
#include "mnist.hpp"
#include "random.hpp"

void invoke_mnist_loader(int rank, int last, int epochs, int batchsize) {
  MPI_Group world_group;
  MPI_Comm_group(MPI_COMM_WORLD, &world_group);

  const int next_ranks[2] = {rank, rank + 1};

  MPI_Group next_group;
  MPI_Group_incl(world_group, 2, next_ranks, &next_group);

  MPI_Comm next_comm;
  MPI_Comm_create_group(MPI_COMM_WORLD, next_group, 0, &next_comm);

  /* Broadcast batchsize */
  MPI_Bcast(&batchsize, 1, MPI_INT, rank, MPI_COMM_WORLD);

  /* Setup dataset */
  MNIST<float> mnist("mnist/");

  mnist.train_images.scale(1.0 / 255.0);
  mnist.test_images.scale(1.0 / 255.0);

  int length = mnist.train_images.length;
  int* perm = new int[mnist.train_images.length];

  for(int i = 0; i < length; ++i) {
    perm[i] = i;
  }

  int rows = mnist.train_images.rows;
  int cols = mnist.train_images.cols;
  int dims = rows * cols;
  int size = 10;

  /* Send input/label size */
  MPI_Send(&dims, 1, MPI_INT, 1, 0, next_comm);
  MPI_Send(&size, 1, MPI_INT, last, 0, MPI_COMM_WORLD);

  int images_size = dims * batchsize;
  int labels_size = size * batchsize;

  MPI_Request barrier_request;
  MPI_Status barrier_status;

  /* Start iteration */
  for(int epoch = 0; epoch < epochs; ++epoch) {
    shuffle(perm, perm + length);
    mnist.reorder(perm);

    for(int i = 0; i < mnist.train_length; i += batchsize) {
      float* images = mnist.train_images.getBatch(i, batchsize);
      float* labels = mnist.train_labels.getBatch(i, batchsize);

      Eigen::Map<Eigen::MatrixXf> x(images, batchsize, dims);
      Eigen::Map<Eigen::MatrixXf> y(labels, batchsize, size);

      MPI_Ibarrier(next_comm, &barrier_request);
      MPI_Wait(&barrier_request, &barrier_status);

      MPI_Send(images, images_size, MPI_FLOAT, 1, 0, next_comm);
      MPI_Send(labels, labels_size, MPI_FLOAT, last, 0, MPI_COMM_WORLD);

      delete[] images;
      delete[] labels;
    }
  }

  delete[] perm;

  int halt = 1;

  MPI_Request halt_request;
  MPI_Status halt_status;

  MPI_Ibcast(&halt, 1, MPI_INT, rank, MPI_COMM_WORLD, &halt_request);
  MPI_Wait(&halt_request, &halt_status);

  MPI_Ibcast(&halt, 1, MPI_INT, last, MPI_COMM_WORLD, &halt_request);
  MPI_Wait(&halt_request, &halt_status);

  MPI_Group_free(&world_group);
  MPI_Group_free(&next_group);
  MPI_Comm_free(&next_comm);
}

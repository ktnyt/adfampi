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
  int epochs = 1;
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
  int cols = rank == last ? n_output : n_hidden;
  int cats = n_output;

  int input_count = batchsize * rows;
  int output_count = batchsize * cols;
  int error_count = batchsize * cats;
  int label_count = batchsize * cats;

  float stdW = 1. / sqrt(static_cast<float>(rows));
  float stdU = 1. / sqrt(static_cast<float>(cols));
  float stdB = 1. / sqrt(static_cast<float>(cats));

  Normal genW(0, stdW);
  Normal genU(0, stdU);
  Normal genB(0, stdB);

  MatrixXf W(rows, cols);
  MatrixXf U(cols, rows);
  MatrixXf B(cats, cols);
  VectorXf b = VectorXf::Zero(cols);
  VectorXf c = VectorXf::Zero(rows);


  for(int j = 0; j < cols; ++j) {
    for(int i = 0; i < rows; ++i) {
      W(i, j) = genW();
      U(j, i) = genU();
    }
    for(int i = 0; i < cats; ++i) {
      B(i, j) = genB();
    }
  }

  int n_steps = epochs * (n_train / batchsize);

  std::queue<MatrixXf> x_queue;
  std::queue<MatrixXf> y_queue;
  std::queue<MatrixXf> t_queue;

  float loss = 0.;
  float acc = 0.;

  MPI_Status status;
  MPI_Request request = MPI_REQUEST_NULL;

  Eigen::MatrixXf x(batchsize * batchsize, rows);
  Eigen::MatrixXf y(batchsize * batchsize, cols);

  y.data()[0] = 0;

  int steps = 4;

  for(int step = 0; step < steps + last; ++step) {
    MPI_Barrier(MPI_COMM_WORLD);

    if(rank == root) {
      std::cout << "Step: " << step << std::endl;
    } else {
      usleep(1000 * 10);
    }

    if(rank < step + 2 && step < steps + rank - 1) {
      if(rank != root) {
        std::cout << rank << " recv" << std::endl;
        MPI_Recv(x.data(), x.size(), MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, &status);
      }
    }

    if(rank < step + 1 && step < steps + rank) {
      if(rank == root) {
        float* image = mnist.train_images.getBatch(step * batchsize, batchsize);
        delete[] image;
      }

      if(rank != last) {
        MPI_Send(y.data(), y.size(), MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD);
        std::cout << rank << " send " << std::endl;
        y.data()[0] += 1;
      }
    }
  }

  std::cout << rank << " done" << std::endl;

  delete[] perm;
  
  MPI_Finalize();

  return 0;
}

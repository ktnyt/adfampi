#include <iostream>
#include <iomanip>
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

  int steps = epochs * (n_train / batchsize);
  int batch = 0;

  std::queue<MatrixXf> x_queue;
  std::queue<MatrixXf> y_queue;
  std::queue<MatrixXf> t_queue;

  float loss = 0.;
  float acc = 0.;

  MPI_Status status;
  MPI_Request request = MPI_REQUEST_NULL;

  for(int step = 0; step < steps + last; ++step) {
    MatrixXf x(batchsize, rows);
    MatrixXf y(batchsize, cols);

    // Load phase
    if(step < steps) {
      if(rank == root) {
        float* images = mnist.train_images.getBatch(step * batchsize, batchsize);
        float* labels = mnist.train_labels.getBatch(step * batchsize, batchsize);
        std::copy(images, images + x.size(), x.data());
        MPI_Send(labels, label_count, MPI_FLOAT, last, 0, MPI_COMM_WORLD);
        delete[] images;
        delete[] labels;
      }

      if(rank == last) {
        MatrixXf t(batchsize, cats);
        MPI_Recv(t.data(), t.size(), MPI_FLOAT, root, 0, MPI_COMM_WORLD, &status);
        t_queue.push(t);
      }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Receive phase
    if(rank < step + 2 && step < steps + rank - 1) {
      if(rank != root) {
        MPI_Recv(x.data(), x.size(), MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, &status);
      }
    }

    // Forward phase
    if(rank < step + 1 && step < steps + rank) {
      MatrixXf a = x * W;
      a.transpose().colwise() += b;

      if(rank != last) {
        y.noalias() = a.unaryExpr(&sigmoid);
        MPI_Send(y.data(), y.size(), MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD);
      } else {
        for(int i = 0; i < batchsize; ++i) {
          y.row(i) = softmax(a.row(i));
        }
      }

      x_queue.push(x);
      y_queue.push(y);
    }

    MatrixXf e(batchsize, cats);

    if(step > last - 1) {
      if(rank == last) {
        MatrixXf t = t_queue.front();
        t_queue.pop();

        e = y - t;

        loss += cross_entropy(y, t);
        acc += accuracy(y, t);
        batch += 1;

        if(batch % 8 == 0) {
          std::cerr << "#" << std::flush;
        }

        if((batch * batchsize) % n_train == 0) {
          loss /= batch;
          acc /= batch;

          std::cerr << std::endl << loss << " " << acc << std::endl;

          loss = 0.F;
          acc = 0.F;
          batch = 0;
        }
      }

      MPI_Bcast(e.data(), e.size(), MPI_FLOAT, last, MPI_COMM_WORLD);

      MatrixXf d_x(batchsize, cols);
      MatrixXf d_W(rows, cols);

      Eigen::MatrixXf x = x_queue.front();
      Eigen::MatrixXf y = y_queue.front();
      x_queue.pop();
      y_queue.pop();

      if(rank == last) {
        d_x = e;
      } else {
        d_x = (e * B).array() * y.unaryExpr(&dsigmoid).array();
      }

      d_W = -x.transpose() * d_x;

      W += d_W * lr;

      for(int i = 0; i < d_x.cols(); ++i) {
        b(i) += d_x.col(i).sum() * lr;
      }
    }
  }

  std::cout << rank << " done" << std::endl;

  delete[] perm;
  
  MPI_Finalize();

  return 0;
}

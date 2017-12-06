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
  int cols = rank == last ? n_hidden : n_output;
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

  int i = 0;

  for(int step = 0; step < n_steps + last; ++step) {
    if(rank == root) std::cout << step << std::endl;

    if((step * batchsize) % n_train == 0) {
      if(rank == root) {
        shuffle(perm, perm + n_train);
        mnist.reorder(perm);
        i = 0;
      }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Handle labels
    if(step < n_steps) {
      if(rank == root) {
        float* label = mnist.train_labels.getBatch(i, batchsize);
        MPI_Send(label, label_count, MPI_FLOAT, last, 0, MPI_COMM_WORLD);
        std::cout << "send label" << std::endl;
        delete[] label;
      }

      if(rank == last) {
        MatrixXf t(batchsize, cats);
        MPI_Recv(t.data(), label_count, MPI_FLOAT, root, 0, MPI_COMM_WORLD, &status);
        std::cout << "recv label" << std::endl;
        t_queue.push(t);
      }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    Eigen::MatrixXf x(batchsize, rows);
    Eigen::MatrixXf y(batchsize, cols);

    // Handle receive
    if(rank < step + 1 && step - rank < n_steps) {
      if(rank != root) {
        MPI_Recv(x.data(), input_count, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, &status);
        std::cout << rank << ": recv" << std::endl;
      } else {
        float* input = mnist.train_images.getBatch(i, batchsize);
        std::copy(input, input + input_count, x.data());
        delete[] input;
      }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    Eigen::MatrixXf a = x * W;
    a.transpose().colwise() += b;

    if(rank != last) {
      y = a.unaryExpr(&sigmoid);
    } else {
      for(int j = 0; j < batchsize; ++j) {
        y.row(i) = softmax(a.row(i));
      }
    }

    x_queue.push(x);
    y_queue.push(y);

    if(rank < step + 1 && step - rank < n_steps) {
      if(rank != last) {
        MPI_Send(y.data(), output_count, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD);
        std::cout << rank << ": send" << std::endl;
      }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if(step + 1 > last) {
      x = x_queue.front();
      y = y_queue.front();
      x_queue.pop();
      y_queue.pop();

      MatrixXf e(batchsize, cats);

      if(rank == last) {
        Eigen::MatrixXf t = t_queue.front();
        t_queue.pop();
        e = y - t;

        loss += cross_entropy(y, t) * batchsize;
        acc += accuracy(y, t) * batchsize;

        if((step * batchsize) % n_train == 0) {
          loss /= batchsize;
          acc /= batchsize;

          std::cout << loss << "\t" << acc << std::endl;

          loss = 0.;
          acc = 0.;
        }
      }

      MPI_Bcast(e.data(), error_count, MPI_FLOAT, last, MPI_COMM_WORLD);
      std::cout << rank << ": bcast" << std::endl;

      MatrixXf d_x(batchsize, cols);
      MatrixXf d_W(rows, cols);

      if(rank == last) {
        d_x = e;
        d_W = -x.transpose() * e;
      } else {
        d_x = (e * B).array() * y.unaryExpr(&dsigmoid).array();
        d_W = -x.transpose() * d_x;
      }

      W += d_W * lr;

      for(int j = 0; j < d_x.cols(); ++j) {
        b(i) += d_x.col(i).sum() * lr;
      }
    }

    i += batchsize;
  }

  delete[] perm;
  
  MPI_Finalize();

  return 0;
}

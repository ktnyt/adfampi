#include "output_layer.hpp"

#include <iostream>
#include <queue>
#include <cmath>
#include "mpi.h"
#include "Eigen/Core"
#include "random.hpp"
#include "activations.hpp"

Eigen::VectorXf softmax(Eigen::VectorXf v) {
  Eigen::VectorXf max = Eigen::VectorXf::Ones(v.size()) * v.maxCoeff();
  Eigen::VectorXf r = (v - max).unaryExpr(&expf);
  return r / r.sum();
}

float accuracy(Eigen::MatrixXf y, Eigen::MatrixXf t) {
  float total = 0.0f;
  for(std::size_t i = 0; i < y.rows(); ++i) {
    Eigen::MatrixXf::Index max_y, max_t, tmp;
    y.row(i).maxCoeff(&tmp, &max_y);
    t.row(i).maxCoeff(&tmp, &max_t);
    if(max_y == max_t) {
      total += 1.0;
    }
  }
  return total / y.rows();
}

float cross_entropy(Eigen::MatrixXf y, Eigen::MatrixXf t) {
  return -(y.array() * ((t.array() + 1e-10f).log())).sum() / t.rows();
}

void invoke_output_layer(int rank, int root, int n_output, float lr) {
  MPI_Group world_group;
  MPI_Comm_group(MPI_COMM_WORLD, &world_group);

  const int root_ranks[1] = {root};

  MPI_Group root_group;
  MPI_Group_incl(world_group, 1, root_ranks, &root_group);

  MPI_Group layer_group;
  MPI_Group_difference(world_group, root_group, &layer_group);

  MPI_Comm layer_comm;
  MPI_Comm_create_group(MPI_COMM_WORLD, layer_group, 0, &layer_comm);

  const int prev_ranks[2] = {rank - 1, rank};

  MPI_Group prev_group;
  MPI_Group_incl(world_group, 2, prev_ranks, &prev_group);

  MPI_Comm prev_comm;
  MPI_Comm_create_group(MPI_COMM_WORLD, prev_group, 0, &prev_comm);

  /* Receive batchsize and input_label size */
  int batchsize;
  int n_input;
  int n_label;
  MPI_Status n_input_status;
  MPI_Status n_label_status;

  MPI_Bcast(&batchsize, 1, MPI_INT, root, MPI_COMM_WORLD);
  MPI_Recv(&n_label, 1, MPI_INT, root, 0, MPI_COMM_WORLD, &n_label_status);
  MPI_Recv(&n_input, 1, MPI_INT, 0, 0, prev_comm, &n_input_status);
  MPI_Bcast(&n_output, 1, MPI_INT, rank - 1, layer_comm);

  /* Listen for the halt signal */
  int halt_loader = 0;
  int halt_output = 1;
  MPI_Request halt_loader_request;
  MPI_Request halt_output_request;
  MPI_Status halt_loader_status;
  MPI_Status halt_output_status;

  MPI_Ibcast(&halt_loader, 1, MPI_INT, root, MPI_COMM_WORLD, &halt_loader_request);

  /* Setup to receive input */
  int input_size = n_input * batchsize;
  float* input;
  MPI_Request input_request = MPI_REQUEST_NULL;
  MPI_Status input_status;
  int input_call = 0;
  int input_flag = 0;

  /* Setup to receive label */
  int label_size = n_label * batchsize;
  float* label;
  MPI_Request label_request = MPI_REQUEST_NULL;
  MPI_Status label_status;
  int label_call = 0;
  int label_flag = 0;

  /* Setup to send output */
  int output_size = n_output * batchsize;
  MPI_Request output_request;
  MPI_Status output_status;

  /* Setup for barrier */
  MPI_Request barrier_request = MPI_REQUEST_NULL;
  MPI_Status barrier_status;
  int barrier_flag;
  int ready = false;

  /* Setup layer */
  std::size_t i, j;

  float stdW = 1. / sqrt(static_cast<float>(n_input));

  Uniform genW(-stdW, stdW);

  Eigen::MatrixXf W(n_input, n_output);
  Eigen::VectorXf b(n_output);

  for(j = 0; j < n_output; ++j) {
    for(i = 0; i < n_input; ++i) {
      W(i, j) = genW();
    }
    b(j) = 0.0;
  }

  /* Setup buffers */
  std::queue<float*> label_queue;

  int batch = 0;
  float acc = 0.0;
  float loss = 0.0;

  while(!halt_loader || label_queue.size() > 0) {
    if(barrier_request == MPI_REQUEST_NULL && !ready) {
      MPI_Ibarrier(prev_comm, &barrier_request);
    }

    MPI_Test(&barrier_request, &barrier_flag, &barrier_status);

    if(barrier_flag) {
      ready = true;
    }

    if(ready) {
      if(input_request == MPI_REQUEST_NULL) {
        input = new float[input_size];
        MPI_Irecv(input, input_size, MPI_FLOAT, 0, 0, prev_comm, &input_request);
      }

      MPI_Test(&input_request, &input_flag, &input_status);

      if(input_flag) {
        Eigen::Map<Eigen::MatrixXf> x(input, batchsize, n_input);
        Eigen::MatrixXf a = (x * W);
        a.transpose().colwise() += b;
        Eigen::MatrixXf y = a:

        for(i = 0; i < batchsize; ++i) {
          y.row(i) = softmax(y.row(i));
        }

        float* label = label_queue.front();
        label_queue.pop();

        Eigen::Map<Eigen::MatrixXf> t(label, batchsize, n_output);
        Eigen::MatrixXf e = y - t;

        loss += cross_entropy(y, t);
        acc += accuracy(y, t);
        batch += 1;

        if(batch % 8 == 0) {
          std::cerr << "#" << std::flush;
        }

        if(batch % 600 == 0) {
          loss /= batch;
          acc /= batch;

          std::cerr << std::endl;
          std::cout << loss << " " << acc << std::endl;

          loss = 0.0;
          acc = 0.0;
          batch = 0;
        }

        MPI_Ibcast(e.data(), output_size, MPI_FLOAT, rank - 1, layer_comm, &output_request);

        MPI_Wait(&output_request, &output_status);

        Eigen::MatrixXf d_W = -x.transpose() * e;

        W += d_W * lr;

        for(i = 0; i < e.cols(); ++i) {
          b(i) += e.col(i).sum();
        }

        delete[] input;
        delete[] label;

        ready = false;
      }
    }

    if(label_request == MPI_REQUEST_NULL) {
      label = new float[label_size];
      MPI_Irecv(label, label_size, MPI_FLOAT, root, 0, MPI_COMM_WORLD, &label_request);
    }

    MPI_Test(&label_request, &label_flag, &label_status);

    if(label_flag) {
      label_queue.push(label);
    }
  }

  MPI_Wait(&halt_loader_request, &halt_loader_status);
  MPI_Ibcast(&halt_output, 1, MPI_INT, rank, MPI_COMM_WORLD, &halt_output_request);
  MPI_Wait(&halt_output_request, &halt_output_status);

  MPI_Group_free(&world_group);
  MPI_Group_free(&layer_group);
  MPI_Group_free(&prev_group);
  MPI_Comm_free(&layer_comm);
  MPI_Comm_free(&prev_comm);
}

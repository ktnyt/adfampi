#include "hidden_layer.hpp"

#include <iostream>
#include <queue>
#include <cmath>
#include "mpi.h"
#include "Eigen/Core"
#include "random.hpp"

float sigmoid(float x) {
  return tanh(x * 0.5f) * 0.5f + 0.5f;
}

float dsigmoid(float y) {
  return y * (1.0f - y);
}

float mean_squared_error(Eigen::MatrixXf y, Eigen::MatrixXf t) {
  Eigen::MatrixXf d = y - t;
  Eigen::VectorXf f(Eigen::Map<Eigen::VectorXf>(d.data(), d.cols()*d.rows()));
  return f.dot(f) / d.size();
}

void invoke_hidden_layer(int rank, int root, int last, int n_output, float lr, float decay) {
  float aelr = lr;

  /* Setup communicators */
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
  const int next_ranks[2] = {rank, rank + 1};

  MPI_Group prev_group;
  MPI_Group next_group;
  MPI_Group_incl(world_group, 2, prev_ranks, &prev_group);
  MPI_Group_incl(world_group, 2, next_ranks, &next_group);

  MPI_Comm prev_comm;
  MPI_Comm next_comm;
  MPI_Comm_create_group(MPI_COMM_WORLD, prev_group, 0, &prev_comm);
  MPI_Comm_create_group(MPI_COMM_WORLD, next_group, 0, &next_comm);

  /* Receive batchsize and input/output size, */
  int batchsize;
  int n_input;
  int n_final;
  MPI_Status n_input_status;

  MPI_Bcast(&batchsize, 1, MPI_INT, root, MPI_COMM_WORLD);
  MPI_Recv(&n_input, 1, MPI_INT, 0, 0, prev_comm, &n_input_status);
  MPI_Send(&n_output, 1, MPI_INT, 1, 0, next_comm);
  MPI_Bcast(&n_final, 1, MPI_INT, last - 1, layer_comm);

  /* Listen for the halt signals */
  int halt_loader = 0;
  int halt_output = 0;
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

  /* Setup to receive error */
  int error_size = n_final * batchsize;
  float* error;
  MPI_Request error_request = MPI_REQUEST_NULL;
  MPI_Status error_status;
  int error_call = 0;
  int error_flag = 0;

  /* Setup to send output */
  int output_size = n_output * batchsize;
  float* output;

  /* Setup for barrier */
  MPI_Request barrier_request = MPI_REQUEST_NULL;
  MPI_Status barrier_status;
  int barrier_flag;
  int ready = false;

  /* Setup layer */
  std::size_t i, j;

  float stdW = 1. / sqrt(static_cast<float>(n_input));
  float stdB = 1. / sqrt(static_cast<float>(n_final));

  Normal genW(0.0, stdW);
  Normal genB(0.0, stdB);

  Eigen::MatrixXf W(n_input, n_output);
  Eigen::MatrixXf B(n_final, n_output);
  Eigen::VectorXf b(n_output);
  Eigen::VectorXf c(n_input);

  for(j = 0; j < n_output; ++j) {
    for(i = 0; i < n_input; ++i) {
      W(i, j) = genW();
      if(j == 0) c(i) = 0.0;
    }
    for(i = 0; i < n_final; ++i) {
      B(i, j) = genB();
    }
    b(j) = 0.0;
  }

  /* Setup buffers */
  std::queue<float*> input_queue;
  std::queue<float*> output_queue;

  while(!(halt_loader && halt_output) || input_queue.size() > 0) {
    if(halt_loader && halt_output_request != MPI_REQUEST_NULL) {
      MPI_Ibcast(&halt_output, 1, MPI_INT, last, MPI_COMM_WORLD, &halt_output_request);
    }

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
        output = new float[output_size];
        MPI_Irecv(input, input_size, MPI_FLOAT, 0, 0, prev_comm, &input_request);
      }

      MPI_Test(&input_request, &input_flag, &input_status);

      if(input_flag) {
        Eigen::Map<Eigen::MatrixXf> x(input, batchsize, n_input);
        Eigen::Map<Eigen::MatrixXf> y(output, batchsize, n_output);
        Eigen::MatrixXf a = (x * W);
        a.transpose().colwise() += b;

        y.noalias() = a.unaryExpr(&sigmoid);

        {
          Eigen::MatrixXf t = y * W.transpose();
          t.transpose().colwise() += c;
          Eigen::MatrixXf z = t.unaryExpr(&sigmoid);

          Eigen::MatrixXf d_z = z - x;
          Eigen::MatrixXf d_y = (d_z * W).array() * y.unaryExpr(&dsigmoid).array();

          Eigen::MatrixXf d_W = -(x.transpose() * d_y + d_z.transpose() * y);
  
          W += d_W * aelr;

          for(i = 0; i < d_y.cols(); ++i) {
            b(i) += d_y.col(i).sum() * aelr;
          }

          for(i = 0; i < d_z.cols(); ++i) {
            c(i) += d_z.col(i).sum() * aelr;
          }

          aelr *= decay;
        }

        input_queue.push(input);
        output_queue.push(output);

        MPI_Ibarrier(next_comm, &barrier_request);
        MPI_Wait(&barrier_request, &barrier_status);
        MPI_Send(output, output_size, MPI_FLOAT, 1, 0, next_comm);

        ready = false;
      }
    }

    if(error_request == MPI_REQUEST_NULL) {
      error = new float[error_size];
      MPI_Ibcast(error, error_size, MPI_FLOAT, last - 1, layer_comm, &error_request);
    }

    MPI_Test(&error_request, &error_flag, &error_status);

    if(error_flag) {
      float* input = input_queue.front();
      float* output = output_queue.front();

      input_queue.pop();
      output_queue.pop();

      Eigen::Map<Eigen::MatrixXf> x(input, batchsize, n_input);
      Eigen::Map<Eigen::MatrixXf> y(output, batchsize, n_output);
      Eigen::Map<Eigen::MatrixXf> e(error, batchsize, n_final);

      Eigen::MatrixXf d_x = (e * B).array() * y.unaryExpr(&dsigmoid).array();
      Eigen::MatrixXf d_W = -x.transpose() * d_x;

      W += d_W * lr;

      for(i = 0; i < d_x.cols(); ++i) {
        b(i) += d_x.col(i).sum() * lr;
      }

      delete[] input;
      delete[] output;
      delete[] error;
    }
  }

  MPI_Group_free(&world_group);
  MPI_Group_free(&layer_group);
  MPI_Group_free(&prev_group);
  MPI_Group_free(&next_group);
  MPI_Comm_free(&layer_comm);
  MPI_Comm_free(&prev_comm);
  MPI_Comm_free(&next_comm);
}

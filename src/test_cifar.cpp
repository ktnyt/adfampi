#include <iostream>
#include <vector>
#include "Eigen/Core"
#include "cifar.hpp"
#include "activations.hpp"
#include "utils.hpp"

int main(int argc, char* argv[]) {
  if(argc < 1) {
    return -1;
  }

  CIFAR<float> cifar("cifar");
  cifar.train->scale(1.0 / 255.0);
  cifar.test->scale(1.0 / 255.0);

  int n_layers = atoi(argv[1]);

  std::vector<Eigen::MatrixXf> Ws;
  std::vector<Eigen::VectorXf> bs;

  for(int i = 0; i < n_layers; ++i) {
    Eigen::MatrixXf W;
    Eigen::VectorXf b;

    deserialize_matrix(format_name(i + 1, "W").c_str(), W);
    deserialize_vector(format_name(i + 1, "b").c_str(), b);

    Ws.push_back(W);
    bs.push_back(b);
  }

  int batchsize = 100;

  {
    float loss = 0.0;
    float acc = 0.0;

    for(int i = 0; i < cifar.train->length; i += batchsize) {
      float* images = cifar.train->getImageBatch(i, batchsize);
      float* labels = cifar.train->getLabelBatch(i, batchsize);

      Eigen::Map<Eigen::MatrixXf> x(images, batchsize, cifar.train->dims);
      Eigen::Map<Eigen::MatrixXf> t(labels, batchsize, 10);

      Eigen::MatrixXf h = x;

      for(int j = 0; j < n_layers; ++j) {
        Eigen::MatrixXf a = h * Ws[j];
        a.transpose().colwise() += bs[j];
        if(j + 1 < n_layers) {
          h = a.unaryExpr(&sigmoid);
        } else {
          h = a;
        }
      }

      for(int j = 0; j < batchsize; ++j) {
        h.row(j) = softmax(h.row(j));
      }

      loss += cross_entropy(h, t) * batchsize;
      acc += accuracy(h, t) * batchsize;
    }

    std::cout << "Loss: " << loss / cifar.train->length << std::endl;
    std::cout << "Accuracy: " << acc / cifar.train->length << std::endl;
  }

  {
    float loss = 0.0;
    float acc = 0.0;

    for(int i = 0; i < cifar.test->length; i += batchsize) {
      float* images = cifar.test->getImageBatch(i, batchsize);
      float* labels = cifar.test->getLabelBatch(i, batchsize);

      Eigen::Map<Eigen::MatrixXf> x(images, batchsize, cifar.test->dims);
      Eigen::Map<Eigen::MatrixXf> t(labels, batchsize, 10);

      Eigen::MatrixXf h = x;

      for(int j = 0; j < n_layers; ++j) {
        Eigen::MatrixXf a = h * Ws[j];
        a.transpose().colwise() += bs[j];
        if(j + 1 < n_layers) {
          h = a.unaryExpr(&sigmoid);
        } else {
          h = a;
        }
      }

      for(int j = 0; j < batchsize; ++j) {
        h.row(j) = softmax(h.row(j));
      }

      loss += cross_entropy(h, t) * batchsize;
      acc += accuracy(h, t) * batchsize;
    }

    std::cout << "Loss: " << loss / cifar.test->length << std::endl;
    std::cout << "Accuracy: " << acc / cifar.test->length << std::endl;
  }

  return 0;
}

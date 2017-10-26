#include <iostream>
#include <vector>
#include "Eigen/Core"
#include "mnist.hpp"
#include "activations.hpp"
#include "utils.hpp"

int main(int argc, char* argv[]) {
  if(argc < 1) {
    return -1;
  }

  MNIST<float> mnist("mnist");
  mnist.train_images.scale(1.0 / 255.0);
  mnist.test_images.scale(1.0 / 255.0);

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

    for(int i = 0; i < mnist.train_images.length; i += batchsize) {
      float* images = mnist.train_images.getBatch(i, batchsize);
      float* labels = mnist.train_labels.getBatch(i, batchsize);

      Eigen::Map<Eigen::MatrixXf> x(images, batchsize, mnist.train_images.dims);
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

    std::cout << "Loss: " << loss / mnist.train_images.length << std::endl;
    std::cout << "Accuracy: " << acc / mnist.train_images.length << std::endl;
  }

  {
    float loss = 0.0;
    float acc = 0.0;

    for(int i = 0; i < mnist.test_images.length; i += batchsize) {
      float* images = mnist.test_images.getBatch(i, batchsize);
      float* labels = mnist.test_labels.getBatch(i, batchsize);

      Eigen::Map<Eigen::MatrixXf> x(images, batchsize, mnist.test_images.dims);
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

    std::cout << "Loss: " << loss / mnist.test_images.length << std::endl;
    std::cout << "Accuracy: " << acc / mnist.test_images.length << std::endl;
  }

  return 0;
}

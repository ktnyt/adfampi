#ifndef __ADFAMPI_FUNCTIONS_HPP__
#define __ADFAMPI_FUNCTIONS_HPP__

#include "Eigen/Core"

typedef Eigen::MatrixXf (*Function)(Eigen::MatrixXf&);

struct Sigmoid {
  static MatrixXf forward(MatrixXf& x) {
    return x.unaryExpr(&sigmoid);
  }

  static MatrixXf backward(MatrixXf& y) {
    return y.unaryExpr(&dsigmoid);
  }
};

struct Softmax {
  static MatrixXf forward(MatrixXf& x) {
    MatrixXf y(x.rows(), x.cols());
    for(int i = 0; i < x.rows(); ++i) {
      y.row(i) = softmax(x.row(i));
    }
    return y;
  }

  static MatrixXf backward(MatrixXf& y) {
    return MatrixXf::Ones(y.rows(), y.cols());
  }
};

#endif  // __ADFAMPI_FUNCTIONS_HPP__
#include "utils.hpp"

#include <sstream>
#include <string>
#include "Eigen/Core"

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

float mean_squared_error(Eigen::MatrixXf y, Eigen::MatrixXf t) {
  Eigen::MatrixXf d = y - t;
  Eigen::VectorXf f(Eigen::Map<Eigen::VectorXf>(d.data(), d.cols()*d.rows()));
  return f.dot(f) / d.size();
}

std::string format_name(int rank, const char* name) {
  std::ostringstream os;
  os << "layer" << rank << "_" << name;
  return os.str();
}

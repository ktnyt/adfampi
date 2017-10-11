#include "Eigen/Core"

Eigen::VectorXf softmax(Eigen::VectorXf v);
float accuracy(Eigen::MatrixXf y, Eigen::MatrixXf t);
float cross_entropy(Eigen::MatrixXf y, Eigen::MatrixXf t);
void invoke_output_layer(int rank, int root, int n_output, float lr=0.05);

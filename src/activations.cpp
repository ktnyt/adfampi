#include "activations.hpp"
#include <cmath>

float dtanh(float y) {
  return 1.0f - y * y;
}

float sigmoid(float x) {
  return tanh(x * 0.5f) * 0.5f + 0.5f;
}

float dsigmoid(float y) {
  return y * (1.0 - y);
}

float dsigmoidx(float x) {
  return dsigmoid(sigmoid(x));
}

float swish(float x) {
  return x * sigmoid(x);
}

float dswishx(float x) {
  float s = sigmoid(x);
  float y = x * s;
  return y + s * (1.0f - y);
}

float relu(float x) {
  return x > 0 ? x : 0.0f;
}

float drelu(float y) {
  return y > 0 ? 1.0f : 0.0f;
}

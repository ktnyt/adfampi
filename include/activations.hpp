#ifndef __ADFA_ACTIVATIONS_HPP__
#define __ADFA_ACTIVATIONS_HPP__

float dtanh(float y);
float sigmoid(float x);
float dsigmoid(float y);
float dsigmoidx(float x);
float swish(float x);
float dswishx(float x);
float relu(float x);
float drelu(float y);

#endif // __ADFA_ACTIVTOINS_HPP__

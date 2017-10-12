#include "random.hpp"
#include <utility>
#include <cmath>
#include <ctime>
#include <limits>
#include "mt64.h"

void shuffle(int* begin, int* end) {
  int range = end - begin;
  int i, j;

  init_genrand64(time(NULL));

  for(i = range - 1; i > 0; --i) {
    j = static_cast<int>(genrand64_real1() * i);
    std::swap(begin[i], begin[j]);
  }
}

Normal::Normal(float mu, float sigma) : mu(mu), sigma(sigma), has_w(false) {
  init_genrand64(time(NULL));
}

float Normal::operator()() {
  if(has_w) {
    has_w = false;
    return w * sigma + mu;
  }

  do {
    u = genrand64_real3() * 2 - 1.0;
    v = genrand64_real3() * 2 - 1.0;
    s = u * u + v * v;
  } while(s >= 1 || s == 0);

  m = sqrt(-2.0 * log(s) / s);
  w = v * m;
  has_w = true;
  return u * sigma + mu;
}

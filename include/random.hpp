void shuffle(int* begin, int* end);

class Uniform {
public:
  Uniform(float min=0.0, float max=1.0);
  float operator()();
private:
  float min, max;
};

class Normal {
public:
  Normal(float mu=0.0, float sigma=1.0);
  float operator()();
private:
  float mu, sigma, u, v, w, s, m;
  bool has_w;
};

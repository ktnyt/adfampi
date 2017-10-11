void shuffle(int* begin, int* end);

class Normal {
public:
  Normal(float mu=0.0, float sigma=1.0);
  float operator()();
private:
  float mu, sigma, u, v, w, s, m;
  bool has_w;
};

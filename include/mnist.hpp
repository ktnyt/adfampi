#include <iostream>
#include <bitset>
#include <fstream>
#include <sstream>
#include <exception>
#include <stdexcept>
#include <cassert>

template<typename T>
void print_mnist(T* image, T* label) {
  for(std::size_t i = 0; i < 28; ++i) {
    for(std::size_t j = 0; j < 28; ++j) {
      T pixel = image[i * 28 + j];
      if(pixel < 0.25) {
        std::cout << "  ";
      } else if(pixel < 0.5) {
        std::cout << "**";
      } else if(pixel < 0.75) {
        std::cout << "OO";
      } else {
        std::cout << "@@";
      }
    }
    std::cout << std::endl;
  }

  for(std::size_t i = 0; i < 10; ++i) {
    std::cout << i << " " << static_cast<int>(label[i]) << std::endl;
  }
}

template<typename T>
struct MNIST {
  struct Images {
    Images(std::string filename) {
      std::streampos size;
      char* buffer;

      std::ifstream file(filename.c_str(), std::ios::in | std::ios::binary | std::ios::ate);
      std::ostringstream os;

      if(!file.is_open()) {
        os << "Failed to load images: file '" << filename << "' does not exist";
        throw std::runtime_error(os.str());
      }

      size = file.tellg();

      // File must be at least 16 bits long
      if(size < 16) {
        os << "Failed to load images: file '" << filename << "' is too small";
        throw std::runtime_error(os.str());
      }

      // Read the file into memory
      buffer = new char[size];
      file.seekg(0, std::ios::beg);
      file.read(buffer, size);
      file.close();

      // Start parsing the contents
      int position = 0;

      /// Read the magic number
      int magic = 0;

      for(int i = 0; i < 4; ++i, ++position) {
        int mask = 0x000000FF & buffer[position];
        magic |= mask << (3 - i) * 8;
      }

      if(magic != 0x00000803) {
        os << "Failed to load images: wrong magic number '" << magic << "'";
        delete[] buffer;
        throw std::runtime_error(os.str());
      }

      /// Read the length of images
      length = 0;

      for(int i = 0; i < 4; ++i, ++position) {
        int mask = 0x000000FF & buffer[position];
        length |= mask << (3 - i) * 8;
      }

      /// Read the number of rows
      rows = 0;

      for(int i = 0; i < 4; ++i, ++position) {
        int mask = 0x000000FF & buffer[position];
        rows |= mask << (3 - i) * 8;
      }

      /// Read the number of columns
      cols = 0;

      for(int i = 0; i < 4; ++i, ++position) {
        int mask = 0x000000FF & buffer[position];
        cols |= mask << (3 - i) * 8;
      }

      dims = rows * cols;

      if(size != length * dims + position) {
        os << "Failed to load images: file does not have expected length";
        delete[] buffer;
        throw std::runtime_error(os.str());
      }

      data = new T[length * dims];

      for(int i = 0; i < length * dims; ++i, ++position) {
        int pixel = buffer[position] & 0x000000FF;
        data[i] = static_cast<T>(pixel);
      }

      delete[] buffer;
    }

    ~Images() {
      delete[] data;
    }

    void reorder(int* perm) {
      T* tmp = new T[length * dims];

      for(int i = 0; i < length; ++i) {
        T* origin = data + perm[i] * dims;
        T* target = tmp + i * dims;
        std::copy(origin, origin + dims, target);
      }

      delete[] data;

      data = tmp;
    }

    T* operator[](int index) {
      return data + dims * index;
    }

    T* getBatch(int from, int size) {
      T* buffer = new T[size * dims];
      for(int i = 0; i < size; ++i) {
        for(int j = 0; j < dims; ++j) {
          buffer[j * size + i] = data[(from + i) * dims + j];
        }
      }
      return buffer;
    }

    void scale(T v) {
      for(int i = 0; i < length * dims; ++i) {
        data[i] *= v;
      }
    }

    int length;
    int rows;
    int cols;
    int dims;
    T* data;
  };

  struct Labels {
    Labels(std::string filename) {
      std::streampos size;
      char* buffer;

      std::ifstream file(filename.c_str(), std::ios::in | std::ios::binary | std::ios::ate);
      std::ostringstream os;

      if(!file.is_open()) {
        os << "Failed to load labels: file '" << filename << "' does not exist";
        throw std::runtime_error(os.str());
      }

      size = file.tellg();

      // File must be at least 16 bits long
      if(size < 16) {
        os << "Failed to load labels: file '" << filename << "' is too small";
        throw std::runtime_error(os.str());
      }

      // Read the file into memory
      buffer = new char[size];
      file.seekg(0, std::ios::beg);
      file.read(buffer, size);
      file.close();

      // Start parsing the contents
      int position = 0;

      /// Read the magic number
      int magic = 0;

      for(int i = 0; i < 4; ++i, ++position) {
        int mask = 0x000000FF & buffer[position];
        magic |= mask << (3 - i) * 8;
      }

      if(magic != 0x00000801) {
        os << "Failed to load labels: wrong magic number '" << magic << "'";
        delete[] buffer;
        throw std::runtime_error(os.str());
      }

      /// Read the length of labels
      length = 0;

      for(int i = 0; i < 4; ++i, ++position) {
        int mask = 0x000000FF & buffer[position];
        length |= mask << (3 - i) * 8;
      }

      if(size != length + position) {
        os << "Failed to load labels: file does not have expected length";
        delete[] buffer;
        throw std::runtime_error(os.str());
      }

      data = new T[length * 10];

      for(int i = 0; i < length; ++i, ++position) {
        int label = static_cast<int>(buffer[position]);
        for(int j = 0; j < 10; ++j) {
          data[i * 10 + j] = static_cast<T>(label == j ? 1 : 0);
        }
      }

      delete[] buffer;
    }

    ~Labels() {
      delete[] data;
    }

    void reorder(int* perm) {
      T* tmp = new T[length * 10];

      for(int i = 0; i < length; ++i) {
        T* origin = data + perm[i] * 10;
        T* target = tmp + i * 10;
        std::copy(origin, origin + 10, target);
      }

      delete[] data;

      data = tmp;
    }

    T* operator[](int index) {
      return data + index * 10;
    }

    T* getBatch(int from, int size) {
      T* buffer = new T[size * 10];
      for(int i = 0; i < size; ++i) {
        for(int j = 0; j < 10; ++j) {
          buffer[j * size + i] = data[(from + i) * 10 + j];
        }
      }
      return buffer;
    }

    int length;
    T* data;
  };

  MNIST(std::string path)
    : train_images(Images(path + "/train-images-idx3-ubyte"))
    , train_labels(Labels(path + "/train-labels-idx1-ubyte"))
    , test_images(Images(path + "/t10k-images-idx3-ubyte"))
    , test_labels(Labels(path + "/t10k-labels-idx1-ubyte"))
  {
    assert(train_images.length == train_labels.length);
    train_length = static_cast<int>(train_images.length);

    assert(test_images.length == test_labels.length);
    test_length = static_cast<int>(test_images.length);
  }

  void reorder(int* perm) {
    train_images.reorder(perm);
    train_labels.reorder(perm);
  }

  Images train_images;
  Labels train_labels;
  int train_length;

  Images test_images;
  Labels test_labels;
  int test_length;
};

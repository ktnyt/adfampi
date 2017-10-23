#include <iostream>
#include <bitset>
#include <fstream>
#include <sstream>
#include <exception>
#include <stdexcept>
#include <vector>
#include <cassert>

template<typename T>
struct CIFAR {
  struct Batch {
    Batch(std::vector<std::string> files) : dims(3072) {
      int i, j;
      int n = files.size();
      length = 10000 * n;

      images = new T[length * dims];
      labels = new T[length * 10];

      for(i = 0; i < n; ++i) {
        T* images_offset = images + 30720000 * i;
        T* labels_offset = labels + 100000 * i;

        std::string filename = files[i];

        std::streampos size;
        char* buffer;

        std::ifstream file(filename.c_str(), std::ios::in | std::ios::binary | std::ios::ate);
        std::ostringstream os;

        if(!file.is_open()) {
          os << "Failed to load batch: file '" << filename << "' does not exist";
          throw std::runtime_error(os.str());
        }

        size = file.tellg();

        // Check size
        if(size != 30730000) {
          os << "Failed to load batch: file '" << filename << "' has wrong size";
          throw std::runtime_error(os.str());
        }

        // Read the file into memory
        buffer = new char[size];
        file.seekg(0, std::ios::beg);
        file.read(buffer, size);
        file.close();

        // Start parsing the contents
        int position = 0;

        while(position < size) {
          int label = 0x000000FF & buffer[position];

          for(j = 0; j < 10; ++j) {
            *labels_offset = static_cast<T>(label == j ? 1 : 0);
            ++labels_offset;
          }

          ++position;

          for(j = 0; j < 3072; ++j) {
            int pixel = 0x000000FF & buffer[position];
            *images_offset = static_cast<T>(pixel);
            ++images_offset;
            ++position;
          }
        }
      }
    }

    void reorder(int* perm) {
      {
        T* tmp = new T[length * dims];

        for(int i = 0; i < length; ++i) {
          T* origin = images + perm[i] * dims;
          T* target = tmp + i * dims;
          std::copy(origin, origin + dims, target);
        }

        delete[] images;

        images = tmp;
      }

      {
        T* tmp = new T[length * 10];

        for(int i = 0; i < length; ++i) {
          T* origin = images + perm[i] * 10;
          T* target = tmp + i * 10;
          std::copy(origin, origin + 10, target);
        }

        delete[] labels;

        labels = tmp;
      }
    }

    ~Batch() {
      delete[] images;
      delete[] labels;
    }

    T* getImageBatch(int from, int size) {
      T* buffer = new T[size * dims];
      for(int i = 0; i < size; ++i) {
        for(int j = 0; j < dims; ++j) {
          buffer[j * size + i] = images[(from + i) * dims + j];
        }
      }
      return buffer;
    }

    T* getLabelBatch(int from, int size) {
      T* buffer = new T[size * dims];
      for(int i = 0; i < size; ++i) {
        for(int j = 0; j < 10; ++j) {
          buffer[j * size + i] = labels[(from + i) * 10 + j];
        }
      }
      return buffer;
    }

    void scale(T v) {
      for(int i = 0; i < length * dims; ++i) {
        images[i] *= v;
      }
    }

    int length;
    int dims;
    T* images;
    T* labels;
  };

  CIFAR(std::string directory) {
    std::vector<std::string> train_files;
    std::vector<std::string> test_files;

    train_files.push_back(directory + "/data_batch_1.bin");
    train_files.push_back(directory + "/data_batch_2.bin");
    train_files.push_back(directory + "/data_batch_3.bin");
    train_files.push_back(directory + "/data_batch_4.bin");
    train_files.push_back(directory + "/data_batch_5.bin");

    test_files.push_back(directory + "/test_batch.bin");

    train = new Batch(train_files);
    test = new Batch(test_files);
  }

  ~CIFAR() {
    delete train;
    delete test;
  }

  Batch* train;
  Batch* test;
};

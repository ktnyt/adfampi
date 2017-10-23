add_library(mt STATIC mt19937-64.c)

set(MNIST_FILES adfa_mnist.cpp mnist_loader.cpp hidden_layer.cpp output_layer.cpp random.cpp)
add_executable(adfa_mnist ${MNIST_FILES})
target_link_libraries(adfa_mnist blas mt)

set(CIFAR_FILES adfa_cifar.cpp cifar_loader.cpp hidden_layer.cpp output_layer.cpp random.cpp)
add_executable(adfa_cifar ${CIFAR_FILES})
target_link_libraries(adfa_cifar blas mt)

if(UNIX AND NOT APPLE)
  target_link_libraries(adfa_mnist rt)
  target_link_libraries(adfa_cifar rt)
endif()

add_library(mt STATIC mt19937-64.c)
set(ADFA_FILES mnist_loader.cpp cifar_loader.cpp hidden_layer.cpp
               output_layer.cpp random.cpp activations.cpp utils.cpp)
add_library(adfa STATIC ${ADFA_FILES})
target_link_libraries(adfa blas mt)

add_executable(adfa_mnist adfa_mnist.cpp)
target_link_libraries(adfa_mnist adfa)

add_executable(test_mnist test_mnist.cpp)
target_link_libraries(test_mnist adfa)

add_executable(adfa_cifar adfa_cifar.cpp)
target_link_libraries(adfa_cifar adfa)

add_executable(test_cifar test_cifar.cpp)
target_link_libraries(test_cifar adfa)

if(UNIX AND NOT APPLE)
  target_link_libraries(adfa_mnist rt)
  target_link_libraries(test_mnist rt)
  target_link_libraries(adfa_cifar rt)
  target_link_libraries(test_cifar rt)
endif()

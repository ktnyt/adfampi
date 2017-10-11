add_library(mt STATIC mt19937-64.c)

set(CPP_FILES adfa.cpp mnist_loader.cpp hidden_layer.cpp output_layer.cpp random.cpp)
add_executable(adfa ${CPP_FILES})
target_link_libraries(adfa blas mt)

if(UNIX AND NOT APPLE)
  target_link_libraries(adfa rt)
endif()

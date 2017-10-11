set(CPP_FILES adfa.cpp mnist_loader.cpp hidden_layer.cpp output_layer.cpp)
add_executable(adfa ${CPP_FILES})
target_link_libraries(adfa blas)

if(UNIX AND NOT APPLE)
  target_link_libraries(adfa rt)
endif()

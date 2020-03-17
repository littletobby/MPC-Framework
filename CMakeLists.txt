cmake_minimum_required(VERSION 3.15)
project(SMMLF)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")

add_executable(SMMLF main.cpp Constant.cpp Constant.h SocketOnline.cpp SocketOnline.h SocketManager.cpp SocketManager.h Mat.cpp Mat.h Player.cpp Player.h Op.cpp Op.h NeuronMat.cpp NeuronMat.h NN.cpp NN.h BPGraph.cpp BPGraph.h MathOp.cpp MathOp.h IOManager.cpp IOManager.h)

target_link_libraries(SMMLF ws2_32)
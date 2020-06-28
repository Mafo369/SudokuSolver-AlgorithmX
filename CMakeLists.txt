cmake_minimum_required(VERSION 2.8)
project( SudokuImage )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( SudokuImage SudokuImage.cpp ImageProcessing.cpp DancingNode.cpp)
target_link_libraries( SudokuImage ${OpenCV_LIBS} )
add_definitions( "-Wall -g" )


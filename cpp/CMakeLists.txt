cmake_minimum_required(VERSION 3.14)
project(order_book CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# release build: -O2, debug build: -g -fsanitize=address
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra -Wpedantic)

add_executable(order_book
    main.cpp
    order_book.cpp
)

# to build:
#   mkdir build && cd build && cmake .. && make
# to run:
#   ./order_book

cmake_minimum_required(VERSION 3.16)
project(polytof LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS_RELEASE "-Ofast -march=native -s")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

include_directories(third_party src)
find_package(Threads REQUIRED)

add_executable(compile src/compile.cpp)
add_executable(bco src/bco.cpp)
target_link_libraries(compile Threads::Threads)
target_link_libraries(bco Threads::Threads)

foreach(W 1 2 6)
    add_executable(topp${W} src/cpd.cpp)
    target_compile_definitions(topp${W} PRIVATE VEC_WORDS=${W})
    target_link_libraries(topp${W} Threads::Threads)
endforeach()

foreach(W 1 6)
    add_executable(waring${W} src/waring.cpp)
    target_compile_definitions(waring${W} PRIVATE VEC_WORDS=${W})
    target_link_libraries(waring${W} Threads::Threads)
endforeach()

cmake_minimum_required(VERSION 2.6)

project(epic)

set( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_LIST_DIR}/cmake )
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")    #enable c++ 11

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})

find_package(PNG REQUIRED)
include_directories(${PNG_INCLUDE_DIR})

find_package(Boost 1.56 COMPONENTS program_options filesystem python REQUIRED)
include_directories(${BOOST_INCLUDE_DIRS})
link_directories(${BOOST_LIBRARY_DIRS})

find_package(PythonLibs 2.7 REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

find_package(NumPy REQUIRED)
include_directories(${NUMPY_INCLUDE_DIRS})

include_directories(${CMAKE_SOURCE_DIR})
link_directories(${CMAKE_INSTALL_PREFIX}/lib)

file(GLOB HEADERS "cpp/*.hpp" "cpp/*.h")
file(GLOB FILES "cpp/*.cpp" "cpp/*.c")

add_library(epic SHARED ${HEADERS} ${FILES})
target_link_libraries(epic ${PNG_LIBRARY} jpeg lapack)

PYTHON_ADD_MODULE(pyEpicFlow "py_epicflow.cpp")
target_link_libraries(pyEpicFlow Boost::python ${PYTHON_LIBRARIES} epic)

file(GLOB exes "epicflow.cpp")
foreach(exe IN ITEMS ${exes})
  get_filename_component(script_name ${exe} NAME_WE)
  
  # Add executable
  add_executable(${script_name} ${exe})
  target_link_libraries(${script_name}
    ${PROJECT_NAME}
    ${OpenCV_LIBS})
endforeach()

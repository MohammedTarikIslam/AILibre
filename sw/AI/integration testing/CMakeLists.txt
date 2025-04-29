cmake_minimum_required(VERSION 3.10)
project(IntegrationTesting)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

#cpr needed to for http requests
find_package(cpr REQUIRED)
find_package(nlohmann_json REQUIRED)

add_executable(summariserai summariserai.cpp)
target_link_libraries(summariserai PRIVATE cpr::cpr nlohmann_json::nlohmann_json)
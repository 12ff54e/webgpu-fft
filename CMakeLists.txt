cmake_minimum_required(VERSION 3.20)
project(webgpu_fft VERSION 0.1.0 LANGUAGES CXX)
add_library(webgpu_fft INTERFACE)
add_library(webgpu_fft::webgpu_fft ALIAS webgpu_fft)
target_include_directories(webgpu_fft INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(webgpu_fft INTERFACE cxx_std_20)
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
  enable_testing()
  add_executable(generate_fft_cases tests/generate_cases.cpp)
  target_link_libraries(generate_fft_cases PRIVATE webgpu_fft)
  add_executable(test_fft_generator tests/native.cpp)
  target_link_libraries(test_fft_generator PRIVATE webgpu_fft)
  add_test(NAME generator_validation COMMAND test_fft_generator)
  add_test(NAME generate_cases COMMAND generate_fft_cases ${CMAKE_CURRENT_BINARY_DIR}/cases.json)
  add_custom_target(browser_tests ALL
    COMMAND generate_fft_cases ${CMAKE_CURRENT_BINARY_DIR}/cases.json
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/tests/index.html ${CMAKE_CURRENT_BINARY_DIR}/index.html
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_BINARY_DIR}/src
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/tests/benchmark.html ${CMAKE_CURRENT_BINARY_DIR}/benchmark.html
    DEPENDS generate_fft_cases)
endif()
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
configure_package_config_file(cmake/webgpu_fftConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/webgpu_fftConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/webgpu_fft)
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/webgpu_fftConfigVersion.cmake
  VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
install(TARGETS webgpu_fft EXPORT webgpu_fftTargets)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT webgpu_fftTargets NAMESPACE webgpu_fft::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/webgpu_fft)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/webgpu_fftConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/webgpu_fftConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/webgpu_fft)

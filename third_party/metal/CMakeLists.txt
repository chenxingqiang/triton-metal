# Metal后端CMake配置

# 添加头文件路径
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# 检测MLX依赖
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import mlx.core; print('MLX_FOUND')"
    ERROR_QUIET
    OUTPUT_VARIABLE MLX_CHECK_RESULT
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(NOT MLX_CHECK_RESULT STREQUAL "MLX_FOUND")
    message(WARNING "MLX not found. Metal backend will be disabled.")
    return()
endif()

# 检测是否在Apple Silicon上运行
if(NOT APPLE OR (NOT CMAKE_SYSTEM_PROCESSOR MATCHES "arm64" AND NOT CMAKE_OSX_ARCHITECTURES MATCHES "arm64"))
    message(WARNING "Not running on Apple Silicon. Metal backend will be disabled.")
    return()
endif()

# 添加Metal后端源文件
# 目前没有C++源文件，仅有Python
# 但为未来扩展做准备

# 安装Python模块
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/
        DESTINATION ${PYTHON_SITE_PACKAGES}/triton/third_party/metal
        FILES_MATCHING PATTERN "*.py")

# 安装语言扩展
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/language/
        DESTINATION ${PYTHON_SITE_PACKAGES}/triton/language/extra
        FILES_MATCHING PATTERN "*.py")

# 输出成功消息
message(STATUS "Metal backend enabled") 
cmake_minimum_required(VERSION 3.18)

project(cuSVD LANGUAGES CXX CUDA)

# ===== Debug flags（最简单、最稳妥）=====
set(CMAKE_CUDA_FLAGS_DEBUG "-O0 -g -G")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
set(CMAKE_C_FLAGS_DEBUG   "-O0 -g")

# =========================
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

find_package(CUDAToolkit REQUIRED)


# =========================
# include 路径（全局）
# =========================
include_directories(
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
    ${CMAKE_CURRENT_SOURCE_DIR}/test/common
)

# =========================
# 1️⃣ 算法库：src → libcuSVD
# =========================
file(GLOB CU_SVD_SRC
    src/*.cu
)

add_library(cuSVD STATIC
    ${CU_SVD_SRC}
)

target_link_libraries(cuSVD
    CUDA::cudart
    CUDA::cublas
)

# =========================
# 2️⃣ 测试公共代码库：test/common
# =========================
file(GLOB TEST_COMMON_SRC
    test/common/*.cu
)

add_library(test_common STATIC
    ${TEST_COMMON_SRC}
)

target_link_libraries(test_common
    CUDA::cudart
)

# =========================
# 3️⃣ 自动为 test/cases 下每个 .cu 生成可执行文件
# =========================
file(GLOB TEST_CASES
    test/cases/*.cu
)

foreach(test_src ${TEST_CASES})
    # 取文件名作为 executable 名
    get_filename_component(test_name ${test_src} NAME_WE)

    add_executable(${test_name}
        ${test_src}
    )

    target_link_libraries(${test_name}
        cuSVD
        test_common
        CUDA::cudart
        CUDA::cublas
    )

    target_compile_options(${test_name} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    )
endforeach()

FIND_PACKAGE(cuPoly REQUIRED)
# FIND_PACKAGE(spog REQUIRED)


# demo: simple_mul
CUDA_ADD_EXECUTABLE(
    simple_mul
    EXCLUDE_FROM_ALL
    ${CMAKE_CURRENT_SOURCE_DIR}/simple_mul.cpp
)

TARGET_LINK_LIBRARIES(
    simple_mul
    -lcupolybfv
    -lntl 
    -lgmp 
    -fopenmp
    ${CUDA_curand_LIBRARY}
    spog
)

# ADD_DEPENDENCIES(simple_mul simple_mul_target)

# demo: toy_mul
CUDA_ADD_EXECUTABLE(
    toy_mul
    EXCLUDE_FROM_ALL
    ${CMAKE_CURRENT_SOURCE_DIR}/toy_mul.cpp
)

TARGET_LINK_LIBRARIES(
    toy_mul
    -lcupolybfv
    -lntl 
    -lgmp 
    -fopenmp
    ${CUDA_curand_LIBRARY}
    spog
)

# demo: add_mul
CUDA_ADD_EXECUTABLE(
    add_mul
    EXCLUDE_FROM_ALL
    ${CMAKE_CURRENT_SOURCE_DIR}/add_mul.cpp
)

TARGET_LINK_LIBRARIES(
    add_mul
    -lcupolybfv
    -lntl 
    -lgmp 
    -fopenmp
    ${CUDA_curand_LIBRARY}
    spog
)

# demo: sequencial_mul
CUDA_ADD_EXECUTABLE(
    sequential_mul
    EXCLUDE_FROM_ALL
    ${CMAKE_CURRENT_SOURCE_DIR}/sequential_mul.cpp
)

TARGET_LINK_LIBRARIES(
    sequential_mul
    -lcupolybfv
    -lntl 
    -lgmp 
    -fopenmp
    ${CUDA_curand_LIBRARY}
    spog
)

# demo: add_mul
CUDA_ADD_EXECUTABLE(
    encrypt_decrypt
    EXCLUDE_FROM_ALL
    ${CMAKE_CURRENT_SOURCE_DIR}/encrypt_decrypt.cpp
)

TARGET_LINK_LIBRARIES(
    encrypt_decrypt
    -lcupolybfv
    -lntl 
    -lgmp 
    -fopenmp
    ${CUDA_curand_LIBRARY}
    spog
)

# demo: simple_add1
CUDA_ADD_EXECUTABLE(
    simple_add1
    EXCLUDE_FROM_ALL
    ${CMAKE_CURRENT_SOURCE_DIR}/simple_add1.cpp
)

TARGET_LINK_LIBRARIES(
    simple_add1
    -lcupolybfv
    -lntl 
    -lgmp 
    -fopenmp
    ${CUDA_curand_LIBRARY}
    spog
)

# demo: simple_add2
CUDA_ADD_EXECUTABLE(
    simple_add2
    EXCLUDE_FROM_ALL
    ${CMAKE_CURRENT_SOURCE_DIR}/simple_add2.cpp
)

TARGET_LINK_LIBRARIES(
    simple_add2
    -lcupolybfv
    -lntl 
    -lgmp 
    -fopenmp
    ${CUDA_curand_LIBRARY}
    spog
)

# demo: simple_add2
CUDA_ADD_EXECUTABLE(
    simple_mul_with_keys
    EXCLUDE_FROM_ALL
    ${CMAKE_CURRENT_SOURCE_DIR}/simple_mul_with_keys.cpp
)

TARGET_LINK_LIBRARIES(
    simple_mul_with_keys
    -lcupolybfv
    -lntl 
    -lgmp 
    -fopenmp
    ${CUDA_curand_LIBRARY}
    spog
)

# custom target
ADD_CUSTOM_COMMAND(
    OUTPUT demos.out
    DEPENDS simple_mul toy_mul add_mul sequential_mul encrypt_decrypt simple_add1 simple_add2 simple_mul_with_keys
    COMMAND ${CMAKE_COMMAND} -E touch demos.out
    )
ADD_CUSTOM_TARGET(
    demos
    DEPENDS demos.out
    )
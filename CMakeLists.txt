cmake_minimum_required(VERSION 3.0) 
# Define project name
project(HMM)
# add_definitions(-DCVF_TESTER)
message("===== HMM =====") 

# Compiler flags
if (CMAKE_COMPILER_IS_GNUCC)
    set (CMAKE_CXX_FLAGS "-D_REETRANT -Wall -Wextra -pedantic -Wno-long-long")
	if (CMAKE_BUILD_TYPE STREQUAL "Debug")
         set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb -O0 -pg")
         message("-- Debug Mode")
	elseif( CMAKE_BUILD_TYPE STREQUAL "Release" )
		set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g -fno-strict-aliasing")
		message("-- Release Mode")
	endif ()
endif (CMAKE_COMPILER_IS_GNUCC)

# Set C++ configuration
set(CMAKE_CXX_STANDARD 11)
set(CXX_EXTENSIONS ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


set(HMMLIB
    ${PROJECT_SOURCE_DIR}/include/hmm.hpp
    ${PROJECT_SOURCE_DIR}/src/hmm.cpp
    ${PROJECT_SOURCE_DIR}/include/viterbi.hpp
    ${PROJECT_SOURCE_DIR}/src/viterbi.cpp
)

# add_library(LINKED_LIBS STATIC
#     ...
# )


# Add tester of HMM
# Add HMM_tester
add_executable(SimpleHMM_tester ${PROJECT_SOURCE_DIR}/tester/SimpleHMM_tester.cpp ${HMMLIB})
# target_link_libraries(HMM_tester
#     ${LINKED_LIBS}  
#     # nlohmann_json
# )
target_include_directories(SimpleHMM_tester 
PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)


add_executable(HMM_tester ${PROJECT_SOURCE_DIR}/tester/HMM_tester.cpp ${HMMLIB})
# target_link_libraries(HMM_tester
#     ${LINKED_LIBS}  
#     # nlohmann_json
# )
target_include_directories(HMM_tester 
PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)




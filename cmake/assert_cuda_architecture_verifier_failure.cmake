foreach(required IN ITEMS CMAKE_COMMAND_PATH VERIFIER CUOBJDUMP ARTIFACT)
    if(NOT DEFINED ${required} OR "${${required}}" STREQUAL "")
        message(FATAL_ERROR "${required} is required")
    endif()
endforeach()

execute_process(
    COMMAND "${CMAKE_COMMAND_PATH}"
        "-DCUOBJDUMP=${CUOBJDUMP}"
        "-DARTIFACT=${ARTIFACT}"
        "-DMEMBER=forevervalidator-cuda-default-architecture-probe"
        "-DARCHITECTURE=sm_61"
        -P "${VERIFIER}"
    RESULT_VARIABLE verifier_result
    OUTPUT_VARIABLE verifier_stdout
    ERROR_VARIABLE verifier_stderr)
set(verifier_output "${verifier_stdout}\n${verifier_stderr}")
string(REPLACE "\r\n" "\n" verifier_output "${verifier_output}")

if(verifier_result EQUAL 0)
    message(FATAL_ERROR
        "CUDA architecture verifier unexpectedly accepted a failing "
        "cuobjdump:\n${verifier_output}")
endif()

string(FIND "${verifier_output}"
    "cuobjdump failed with exit" diagnostic_offset)
if(diagnostic_offset EQUAL -1)
    message(FATAL_ERROR
        "CUDA architecture verifier did not report the cuobjdump exit "
        "failure:\n${verifier_output}")
endif()

message(STATUS
    "Verified the CUDA artifact inspector rejects nonzero cuobjdump exits")

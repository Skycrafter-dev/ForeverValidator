foreach(required IN ITEMS CUOBJDUMP ARTIFACT MEMBER ARCHITECTURE)
    if(NOT DEFINED ${required} OR "${${required}}" STREQUAL "")
        message(FATAL_ERROR "${required} is required")
    endif()
endforeach()

execute_process(
    COMMAND "${CUOBJDUMP}" --list-elf "${ARTIFACT}"
    RESULT_VARIABLE cuobjdump_result
    OUTPUT_VARIABLE cuobjdump_stdout
    ERROR_VARIABLE cuobjdump_stderr)
set(cuobjdump_output "${cuobjdump_stdout}\n${cuobjdump_stderr}")
string(REPLACE "\r\n" "\n" cuobjdump_output "${cuobjdump_output}")

string(FIND "${cuobjdump_output}" "${MEMBER}" member_offset)
if(member_offset EQUAL -1)
    message(FATAL_ERROR
        "CUDA artifact does not contain ${MEMBER} (cuobjdump exit "
        "${cuobjdump_result}):\n${cuobjdump_output}")
endif()

string(SUBSTRING "${cuobjdump_output}" ${member_offset} -1 member_output)
string(FIND "${member_output}" "${ARCHITECTURE}" architecture_offset)
string(FIND "${member_output}" "\nmember " next_member_offset)
if(architecture_offset EQUAL -1 OR
   (NOT next_member_offset EQUAL -1 AND
    architecture_offset GREATER next_member_offset))
    message(FATAL_ERROR
        "CUDA artifact member ${MEMBER} does not contain real "
        "${ARCHITECTURE} code (cuobjdump exit ${cuobjdump_result}):\n"
        "${cuobjdump_output}")
endif()

message(STATUS
    "Verified real ${ARCHITECTURE} code in ${MEMBER}: ${ARTIFACT}")

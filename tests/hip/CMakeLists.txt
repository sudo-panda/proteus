enable_language(HIP)

file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/lit.cfg.py "
import lit.formats

config.name = 'LIT tests'
config.test_format = lit.formats.ShTest(True)

config.suffixes = ['.cpp']
config.test_source_root = '${CMAKE_CURRENT_SOURCE_DIR}'
config.test_exec_root = '${CMAKE_CURRENT_BINARY_DIR}'
ext = lit_config.params['EXT']
config.substitutions.append(('%ext', ext))
"
)

function(CREATE_TEST exe source)
    add_executable(${exe}.hip ${source})
    set_source_files_properties(${source} PROPERTIES
        LANGUAGE HIP
        OBJECT_DEPENDS ${CMAKE_SOURCE_DIR}/pass/JitPass.cpp
    )
    add_dependencies(${exe}.hip JitPass)

    target_link_libraries(${exe}.hip PUBLIC jit)

    target_compile_options(
        ${exe}.hip
        PUBLIC
        -fpass-plugin=$<TARGET_FILE:JitPass>
    )

    add_test(NAME ${exe}.hip COMMAND ${LIT} -vv -D EXT=hip ${source})
endfunction()

function(CREATE_TEST_RDC exe source)
    add_executable(${exe}.hip.rdc ${source})
    set_source_files_properties(${source} PROPERTIES
        LANGUAGE HIP
        OBJECT_DEPENDS ${CMAKE_SOURCE_DIR}/pass/JitPass.cpp
    )
    add_dependencies(${exe}.hip.rdc JitPass)

    target_link_libraries(${exe}.hip.rdc PUBLIC jit)

    # This is unsupported see: https://gitlab.kitware.com/cmake/cmake/-/issues/23210
    # set_target_properties(${exe}.rdc PROPERTIES HIP_SEPARABLE_COMPILATION on)

    # Add -fgpu-rdc, --hip-link options for compilation and linking to enable RDC.
    target_compile_options(
        ${exe}.hip.rdc
        PUBLIC
        -fgpu-rdc
        -fpass-plugin=$<TARGET_FILE:JitPass>
    )

    target_link_options(${exe}.hip.rdc PUBLIC -fgpu-rdc --hip-link)

    add_test(NAME ${exe}.hip.rdc COMMAND ${LIT} -vv -D EXT=hip.rdc ${source})
endfunction()

CREATE_TEST(kernel kernel.cpp)
CREATE_TEST(kernel_cache kernel_cache.cpp)
CREATE_TEST(kernel_args kernel_args.cpp)
CREATE_TEST(kernels_gvar kernels_gvar.cpp)
CREATE_TEST(kernel_launches kernel_launches.cpp)
CREATE_TEST(kernel_launches_args kernel_launches_args.cpp)
CREATE_TEST(indirect_launcher indirect_launcher.cpp)
CREATE_TEST(daxpy daxpy.cpp)

CREATE_TEST_RDC(kernel kernel.cpp)
CREATE_TEST_RDC(kernel_cache kernel_cache.cpp)
CREATE_TEST_RDC(kernel_args kernel_args.cpp)
CREATE_TEST_RDC(kernels_gvar kernels_gvar.cpp)
CREATE_TEST_RDC(kernel_launches kernel_launches.cpp)
CREATE_TEST_RDC(kernel_launches_args kernel_launches_args.cpp)
CREATE_TEST_RDC(indirect_launcher indirect_launcher.cpp)
CREATE_TEST_RDC(daxpy daxpy.cpp)

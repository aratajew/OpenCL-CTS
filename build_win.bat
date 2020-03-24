@ECHO off
setlocal ENABLEDELAYEDEXPANSION

IF DEFINED ProgramFiles(x86) SET ProgFilesDir=%ProgramFiles(x86)%
IF NOT DEFINED ProgFilesDir SET ProgFilesDir=%ProgramFiles%

rem -------------------------------- Update these to match what's on your PC ------------------------------------------------

SET VCPATH="%ProgFilesDir%\Microsoft Visual Studio\2019\Professional\Common7\IDE\devenv.exe"

SET PATH=%CMAKEPATH%;%PATH%

rem -------------------------------------------------------------------------------------------------------------------------

setlocal ENABLEDELAYEDEXPANSION

call "%VS160COMNTOOLS%\vsvars32.bat"

mkdir build_win
pushd build_win
IF NOT EXIST CLConform.sln (
   echo "Solution file not found, running Cmake"
   cmake -G "Visual Studio 16 2019" ..\.  -DKHRONOS_OFFLINE_COMPILER="khronos_offline_compiler" -DCL_LIBCLCXX_DIR="cl_libclcxx_dir" -DCL_INCLUDE_DIR="C:/work/igc/vpg-compute-neo/third_party/opencl_headers" -DCL_LIB_DIR="C:/Program Files (x86)/inteloneapi/compiler/2021.1-beta03/windows/lib" -DCMAKE_RUNTIME_OUTPUT_DIRECTORY=. -DOPENCL_LIBRARIES=OpenCL
) else (
   echo "Solution file found CLConform.sln "
)

echo Building CLConform.sln...
%VCPATH% CLConform.sln /build


GOTO:EOF

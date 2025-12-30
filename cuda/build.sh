rm -rf output
mkdir output && cd output
cmake -DCMAKE_BUILD_TYPE=Debug ..
make VERBOSE=1
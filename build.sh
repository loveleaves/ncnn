rm -rf build
mkdir -p build
cd build
cmake ..
make -j32
make install
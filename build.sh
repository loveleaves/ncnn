#!/bin/bash
set -e

BUILD_DIR="build"
jCount=32
MODE=$1

if [ ! $MODE ]; then
    MODE="build"
fi

build_project() {
    cd "$BUILD_DIR"
    cmake ..
    make -j${jCount}
}

rebuild_project() {
    if [ -d "$BUILD_DIR" ]; then
        echo "build folder removed: ${BUILD_DIR}."
        rm -rf "$BUILD_DIR"
    fi
    mkdir -p "$BUILD_DIR"
    build_project
}

install_project() {
    rebuild_project
    make install
}

if [ "$MODE" == "install" ]; then
    install_project
elif [ "$MODE" == "rebuild" ]; then
    rebuild_project
else
    build_project
fi
#!/bin/bash
set -e

BASE_DIR=$(cd "$(dirname "$0")";pwd)
BUILD_FOLDER="build"
jCount=32
MODE=$1
DEMO_DIR="$BASE_DIR/demo"
TESTS_FOLDER="build"
TESTS_DIR="$BASE_DIR/${TESTS_FOLDER}"

if [ ! $MODE ]; then
    MODE="build"
fi

download_3rd() {
    local GTEST_DIR="$TESTS_DIR/thirdParty/googletest"
    mkdir -p "$TESTS_DIR/thirdParty"
    # 检查是否已经下载 GoogleTest
    if [ ! -d "$GTEST_DIR" ]; then
        echo "Downloading GoogleTest..."
        git clone -b master --depth=1 https://github.com/google/googletest.git "$GTEST_DIR"
        echo "GoogleTest downloaded to $GTEST_DIR"
    else
        echo "GoogleTest already exists at $GTEST_DIR"
    fi
}

build_project() {
    local current_path=$1
    local build_path="$current_path/$BUILD_FOLDER"
    if [ ! -d "$build_path" ]; then
        mkdir -p "$build_path"
    fi
    cd "$build_path"
    cmake ..
    make -j${jCount}
}

rebuild_project() {
    local current_path=$1
    local build_path="$1/$BUILD_FOLDER"
    if [ -d "$build_path" ]; then
        echo "build folder removed: ${build_path}."
        rm -rf "$build_path"
    fi
    mkdir -p "$build_path"
    build_project "$current_path"
}

build_tests() {
    local current_path=$1
    cd "$current_path"
    cmake ..
    make -j${jCount}
}

install_project() {
    local current_path=$1
    cd "$current_path"
    make install
}

build_demo() {
    local current_path=$1
    cd "$current_path"
    rebuild_project "$current_path"
}

# init
download_3rd

if [ "$MODE" == "install" ]; then
    install_project "$BASE_DIR/$BUILD_FOLDER"
elif [ "$MODE" == "rebuild" ]; then
    rebuild_project "$BASE_DIR"
elif [ "$MODE" == "demo" ]; then
    build_demo "$DEMO_DIR"
elif [ "$MODE" == "test" ]; then
    build_tests "$BASE_DIR/$BUILD_FOLDER/$TESTS_FOLDER"
elif [ "$MODE" == "all" ]; then
    rebuild_project "$BASE_DIR"
    install_project "$BASE_DIR/$BUILD_FOLDER"
    build_demo "$DEMO_DIR"
else
    build_project "$BASE_DIR"
fi
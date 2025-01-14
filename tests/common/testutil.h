// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#ifndef TESTUTIL_H
#define TESTUTIL_H

#include "cpu.h"
#include "layer.h"
#include "mat.h"

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#define TEST_LAYER_DISABLE_AUTO_INPUT_PACKING (1 << 0)
#define TEST_LAYER_DISABLE_AUTO_INPUT_CASTING (1 << 1)
#define TEST_LAYER_DISABLE_GPU_TESTING        (1 << 2)
#define TEST_LAYER_ENABLE_FORCE_INPUT_PACK8   (1 << 3)

void SRAND(int seed);

uint64_t RAND();

float RandomFloat(float a = -1.2f, float b = 1.2f);

void Randomize(ncnn::Mat& m, float a = -1.2f, float b = 1.2f);

ncnn::Mat RandomMat(int w, float a = -1.2f, float b = 1.2f);

ncnn::Mat RandomMat(int w, int h, float a = -1.2f, float b = 1.2f);

ncnn::Mat RandomMat(int w, int h, int c, float a = -1.2f, float b = 1.2f);
#endif // TESTUTIL_H

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

#include "testutil.h"

#include "cpu.h"
#include "layer.h"
#include "mat.h"
#include "prng.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>

static struct prng_rand_t g_prng_rand_state;

void SRAND(int seed)
{
    prng_srand(seed, &g_prng_rand_state);
}

uint64_t RAND()
{
    return prng_rand(&g_prng_rand_state);
}

float RandomFloat(float a, float b)
{
    float random = ((float)RAND()) / (float)uint64_t(-1); //RAND_MAX;
    float diff = b - a;
    float r = random * diff;
    float v = a + r;
    // generate denormal as zero
    if (v < 0.0001 && v > -0.0001)
        v = 0.f;
    return v;
}

void Randomize(ncnn::Mat& m, float a, float b)
{
    for (size_t i = 0; i < m.total(); i++)
    {
        m[i] = RandomFloat(a, b);
    }
}

ncnn::Mat RandomMat(int w, float a, float b)
{
    ncnn::Mat m(w);
    Randomize(m, a, b);
    return m;
}

ncnn::Mat RandomMat(int w, int h, float a, float b)
{
    ncnn::Mat m(w, h);
    Randomize(m, a, b);
    return m;
}

ncnn::Mat RandomMat(int w, int h, int c, float a, float b)
{
    ncnn::Mat m(w, h, c);
    Randomize(m, a, b);
    return m;
}
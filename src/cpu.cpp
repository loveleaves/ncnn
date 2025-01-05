// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "cpu.h"

#include <stdio.h>
#include <string.h>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ncnn {

int cpu_support_arm_neon()
{
    // TODO
    return 0;
}

int cpu_support_arm_vfpv4()
{
    // TODO
    return 0;
}

int cpu_support_arm_asimdhp()
{
    // TODO
    return 0;
}

static int get_cpucount()
{
    // TODO
    return 1;
}

static int g_cpucount = get_cpucount();

int get_cpu_count()
{
    // TODO
    return g_cpucount;
}

static int g_powersave = 0;

int get_cpu_powersave()
{
    // TODO
    return g_powersave;
}

int set_cpu_powersave(int powersave)
{
    // TODO
    (void) powersave;  // Avoid unused parameter warning.
    return -1;
}

int get_omp_num_threads()
{
#ifdef _OPENMP
    return omp_get_num_threads();
#else
    return 1;
#endif
}

void set_omp_num_threads(int num_threads)
{
#ifdef _OPENMP
    omp_set_num_threads(num_threads);
#else
    (void)num_threads;
#endif
}

int get_omp_dynamic()
{
#ifdef _OPENMP
    return omp_get_dynamic();
#else
    return 0;
#endif
}

void set_omp_dynamic(int dynamic)
{
#ifdef _OPENMP
    omp_set_dynamic(dynamic);
#else
    (void)dynamic;
#endif
}

} // namespace ncnn

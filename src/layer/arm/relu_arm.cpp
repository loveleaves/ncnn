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

#include "relu_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

namespace ncnn {

DEFINE_LAYER_CREATOR(ReLU_arm)

int ReLU_arm::forward_inplace(Mat& bottom_top_blob) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;
    int size = w * h;

    if (slope == 0.f)
    {
        #pragma omp parallel for
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            int i = 0;
#if __ARM_NEON
            float32x4_t _zero = vdupq_n_f32(0.f);
            for (; i + 15 < size; i += 16)
            {
#if NCNN_GNU_INLINE_ASM
#if __aarch64__
                asm volatile(
                    "prfm   pldl1keep, [%0, #512]   \n"
                    "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%0] \n"
                    "fmax   v0.4s, v0.4s, %2.4s     \n"
                    "fmax   v1.4s, v1.4s, %2.4s     \n"
                    "fmax   v2.4s, v2.4s, %2.4s     \n"
                    "fmax   v3.4s, v3.4s, %2.4s     \n"
                    "st1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%0], #64 \n"
                    : "=r"(ptr) // %0
                    : "0"(ptr),
                    "w"(_zero) // %2
                    : "memory", "v0", "v1", "v2", "v3");
#else  // __aarch64__
                asm volatile(
                    "pld        [%0, #512]      \n"
                    "vldm       %0, {d0-d7}     \n"
                    "vmax.f32   q0, q0, %q2     \n"
                    "vmax.f32   q1, q1, %q2     \n"
                    "vmax.f32   q2, q2, %q2     \n"
                    "vmax.f32   q3, q3, %q2     \n"
                    "vstm       %0!, {d0-d7}    \n"
                    : "=r"(ptr) // %0
                    : "0"(ptr),
                    "w"(_zero) // %2
                    : "memory", "q0", "q1", "q2", "q3");
#endif // __aarch64__
#else  // NCNN_GNU_INLINE_ASM
                float32x4_t _p0 = vld1q_f32(ptr);
                float32x4_t _p1 = vld1q_f32(ptr + 4);
                float32x4_t _p2 = vld1q_f32(ptr + 8);
                float32x4_t _p3 = vld1q_f32(ptr + 12);
                _p0 = vmaxq_f32(_p0, _zero);
                _p1 = vmaxq_f32(_p1, _zero);
                _p2 = vmaxq_f32(_p2, _zero);
                _p3 = vmaxq_f32(_p3, _zero);
                vst1q_f32(ptr, _p0);
                vst1q_f32(ptr + 4, _p1);
                vst1q_f32(ptr + 8, _p2);
                vst1q_f32(ptr + 12, _p3);
                ptr += 16;
#endif // NCNN_GNU_INLINE_ASM
            }
            for (; i + 7 < size; i += 8)
            {
                float32x4_t _p0 = vld1q_f32(ptr);
                float32x4_t _p1 = vld1q_f32(ptr + 4);
                _p0 = vmaxq_f32(_p0, _zero);
                _p1 = vmaxq_f32(_p1, _zero);
                vst1q_f32(ptr, _p0);
                vst1q_f32(ptr + 4, _p1);
                ptr += 8;
            }
            for (; i + 3 < size; i += 4)
            {
                float32x4_t _ptr = vld1q_f32(ptr);
                _ptr = vmaxq_f32(_ptr, _zero);
                vst1q_f32(ptr, _ptr);
                ptr += 4;
            }
#endif // __ARM_NEON
            for (; i < size; i++)
            {
                *ptr = std::max(*ptr, 0.f);
                ptr++;
            }
        }
    }
    else
    {
        #pragma omp parallel for
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            int i = 0;
#if __ARM_NEON
            float32x4_t _zero = vdupq_n_f32(0.f);
            float32x4_t _slope = vdupq_n_f32(slope);
            for (; i + 15 < size; i += 16)
            {
#if NCNN_GNU_INLINE_ASM
#if __aarch64__
                asm volatile(
                    "prfm   pldl1keep, [%0, #512]   \n"
                    "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%0] \n"
                    "fcmle  v4.4s, v0.4s, #0        \n"
                    "fcmle  v5.4s, v1.4s, #0        \n"
                    "fcmle  v6.4s, v2.4s, #0        \n"
                    "fcmle  v7.4s, v3.4s, #0        \n"
                    "fmul   v8.4s, v0.4s, %2.4s     \n"
                    "fmul   v9.4s, v1.4s, %2.4s     \n"
                    "fmul   v10.4s, v2.4s, %2.4s    \n"
                    "fmul   v11.4s, v3.4s, %2.4s    \n"
                    "bit    v0.16b, v8.16b, v4.16b  \n"
                    "bit    v1.16b, v9.16b, v5.16b  \n"
                    "bit    v2.16b, v10.16b, v6.16b \n"
                    "bit    v3.16b, v11.16b, v7.16b \n"
                    "st1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%0], #64 \n"
                    : "=r"(ptr) // %0
                    : "0"(ptr),
                    "w"(_slope) // %2
                    : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11");
#else  // __aarch64__
                asm volatile(
                    "pld        [%0, #512]      \n"
                    "vldm       %0, {d0-d7}     \n"
                    "vcle.f32   q4, q0, %q2     \n"
                    "vcle.f32   q5, q1, %q2     \n"
                    "vcle.f32   q6, q2, %q2     \n"
                    "vcle.f32   q7, q3, %q2     \n"
                    "vmul.f32   q8, q0, %q3     \n"
                    "vmul.f32   q9, q1, %q3     \n"
                    "vmul.f32   q10, q2, %q3    \n"
                    "vmul.f32   q11, q3, %q3    \n"
                    "vbit.32    q0, q8, q4      \n"
                    "vbit.32    q1, q9, q5      \n"
                    "vbit.32    q2, q10, q6     \n"
                    "vbit.32    q3, q11, q7     \n"
                    "vstm       %0!, {d0-d7}    \n"
                    : "=r"(ptr) // %0
                    : "0"(ptr),
                    "w"(_zero), // %2
                    "w"(_slope) // %3
                    : "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11");
#endif // __aarch64__
#else  // NCNN_GNU_INLINE_ASM
                float32x4_t _p0 = vld1q_f32(ptr);
                float32x4_t _p1 = vld1q_f32(ptr + 4);
                float32x4_t _p2 = vld1q_f32(ptr + 8);
                float32x4_t _p3 = vld1q_f32(ptr + 12);
                uint32x4_t _lemask0 = vcleq_f32(_p0, _zero);
                uint32x4_t _lemask1 = vcleq_f32(_p1, _zero);
                uint32x4_t _lemask2 = vcleq_f32(_p2, _zero);
                uint32x4_t _lemask3 = vcleq_f32(_p3, _zero);
                float32x4_t _ps0 = vmulq_f32(_p0, _slope);
                float32x4_t _ps1 = vmulq_f32(_p1, _slope);
                float32x4_t _ps2 = vmulq_f32(_p2, _slope);
                float32x4_t _ps3 = vmulq_f32(_p3, _slope);
                _p0 = vbslq_f32(_lemask0, _ps0, _p0);
                _p1 = vbslq_f32(_lemask1, _ps1, _p1);
                _p2 = vbslq_f32(_lemask2, _ps2, _p2);
                _p3 = vbslq_f32(_lemask3, _ps3, _p3);
                vst1q_f32(ptr, _p0);
                vst1q_f32(ptr + 4, _p1);
                vst1q_f32(ptr + 8, _p2);
                vst1q_f32(ptr + 12, _p3);
                ptr += 16;
#endif // NCNN_GNU_INLINE_ASM
            }
            for (; i + 7 < size; i += 8)
            {
                float32x4_t _p0 = vld1q_f32(ptr);
                float32x4_t _p1 = vld1q_f32(ptr + 4);
                uint32x4_t _lemask0 = vcleq_f32(_p0, _zero);
                uint32x4_t _lemask1 = vcleq_f32(_p1, _zero);
                float32x4_t _ps0 = vmulq_f32(_p0, _slope);
                float32x4_t _ps1 = vmulq_f32(_p1, _slope);
                _p0 = vbslq_f32(_lemask0, _ps0, _p0);
                _p1 = vbslq_f32(_lemask1, _ps1, _p1);
                vst1q_f32(ptr, _p0);
                vst1q_f32(ptr + 4, _p1);
                ptr += 8;
            }
            for (; i + 3 < size; i += 4)
            {
                float32x4_t _p = vld1q_f32(ptr);
                uint32x4_t _lemask = vcleq_f32(_p, _zero);
                float32x4_t _ps = vmulq_f32(_p, _slope);
                _p = vbslq_f32(_lemask, _ps, _p);
                vst1q_f32(ptr, _p);
                ptr += 4;
            }
#endif // __ARM_NEON
            for (; i < size; i++)
            {
                if (*ptr < 0)
                    *ptr *= slope;
                ptr++;
            }
        }
    }

    return 0;
}

} // namespace ncnn

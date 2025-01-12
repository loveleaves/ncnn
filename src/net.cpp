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

#include "net.h"

#include "datareader.h"
#include "layer_type.h"
#include "modelbin.h"
#include "paramdict.h"

#include <stdio.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif // _OPENMP

#if NCNN_BENCHMARK
#include "benchmark.h"
#endif // NCNN_BENCHMARK

namespace ncnn {

Net::Net()
{
}

Net::~Net()
{
    clear();
}

#if NCNN_STRING
int Net::register_custom_layer(const char* type, layer_creator_func creator)
{
    int typeindex = layer_to_index(type);
    if (typeindex != -1)
    {
        NCNN_LOGE("can not register build-in layer type %s", type);
        return -1;
    }

    int custom_index = custom_layer_to_index(type);
    if (custom_index == -1)
    {
        struct layer_registry_entry entry = { type, creator };
        custom_layer_registry.push_back(entry);
    }
    else
    {
        NCNN_LOGE("overwrite existing custom layer type %s", type);
        custom_layer_registry[custom_index].name = type;
        custom_layer_registry[custom_index].creator = creator;
    }

    return 0;
}
#endif // NCNN_STRING

int Net::register_custom_layer(int index, layer_creator_func creator)
{
    int custom_index = index & ~LayerType::CustomBit;
    if (index == custom_index)
    {
        NCNN_LOGE("can not register build-in layer index %d", custom_index);
        return -1;
    }

    if ((int)custom_layer_registry.size() <= custom_index)
    {
#if NCNN_STRING
        struct layer_registry_entry dummy = { "", 0 };
#else
        struct layer_registry_entry dummy = { 0 };
#endif // NCNN_STRING
        custom_layer_registry.resize(custom_index + 1, dummy);
    }

    if (custom_layer_registry[custom_index].creator)
    {
        NCNN_LOGE("overwrite existing custom layer index %d", custom_index);
    }

    custom_layer_registry[custom_index].creator = creator;
    return 0;
}

#if NCNN_STRING
int Net::load_param(const DataReader& dr)
{
#define SCAN_VALUE(fmt, v)                \
    if (dr.scan(fmt, &v) != 1)            \
    {                                     \
        NCNN_LOGE("parse " #v " failed"); \
        return -1;                        \
    }

    int magic = 0;
    SCAN_VALUE("%d", magic)
    if (magic != 7767517)
    {
        NCNN_LOGE("param is too old, please regenerate");
        return -1;
    }

    // parse
    int layer_count = 0;
    int blob_count = 0;
    SCAN_VALUE("%d", layer_count)
    SCAN_VALUE("%d", blob_count)
    if (layer_count <= 0 || blob_count <= 0)
    {
        NCNN_LOGE("invalid layer_count or blob_count");
        return -1;
    }

    layers.resize((size_t)layer_count);
    blobs.resize((size_t)blob_count);

    ParamDict pd;

    int blob_index = 0;
    for (int layer_index = 0; layer_index < layer_count; layer_index++)
    {
        char layer_type[256];
        char layer_name[256];
        int bottom_count = 0;
        int top_count = 0;
        SCAN_VALUE("%255s", layer_type)
        SCAN_VALUE("%255s", layer_name)
        SCAN_VALUE("%d", bottom_count)
        SCAN_VALUE("%d", top_count)

        Layer* layer = create_layer(layer_type);
        if (!layer)
        {
            layer = create_custom_layer(layer_type);
        }
        if (!layer)
        {
            NCNN_LOGE("layer %s not exists or registered", layer_type);
            clear();
            return -1;
        }

        layer->type = std::string(layer_type);
        layer->name = std::string(layer_name);
        //         NCNN_LOGE("new layer %d %s", i, layer_name);

        layer->bottoms.resize(bottom_count);
        for (int i=0; i<bottom_count; i++)
        {
            char bottom_name[256];
            SCAN_VALUE("%255s", bottom_name)

            int bottom_blob_index = find_blob_index_by_name(bottom_name);
            if (bottom_blob_index == -1)
            {
                Blob& blob = blobs[blob_index];

                bottom_blob_index = blob_index;

                blob.name = std::string(bottom_name);
                //                 NCNN_LOGE("new blob %s", bottom_name);

                blob_index++;
            }

            Blob& blob = blobs[bottom_blob_index];

            blob.consumers.push_back(layer_index);

            layer->bottoms[i] = bottom_blob_index;
        }

        layer->tops.resize(top_count);
        for (int i=0; i<top_count; i++)
        {
            Blob& blob = blobs[blob_index];

            char blob_name[256];
            SCAN_VALUE("%255s", blob_name)

            blob.name = std::string(blob_name);
            //             NCNN_LOGE("new blob %s", blob_name);

            blob.producer = layer_index;

            layer->tops[i] = blob_index;

            blob_index++;
        }

        // layer specific params
        int pdlr = pd.load_param(dr);
        if (pdlr != 0)
        {
            NCNN_LOGE("ParamDict load_param %d %s failed", layer_index, layer_name);
            continue;
        }

        int lr = layer->load_param(pd);
        if (lr != 0)
        {
            NCNN_LOGE("layer load_param failed");
            continue;
        }

        layers[layer_index] = layer;
    }

#undef SCAN_VALUE
    return 0;
}
#endif // NCNN_STRING

int Net::load_param_bin(const DataReader& dr)
{
#if __BIG_ENDIAN__
#define READ_VALUE(buf)                            \
    if (dr.read(&buf, sizeof(buf)) != sizeof(buf)) \
    {                                              \
        NCNN_LOGE("read " #buf " failed");         \
        return -1;                                 \
    }                                              \
    if (sizeof(buf) == 2)                          \
        swap_endianness_16(&buf);                  \
    if (sizeof(buf) == 4)                          \
        swap_endianness_32(&buf);
#else
#define READ_VALUE(buf)                            \
    if (dr.read(&buf, sizeof(buf)) != sizeof(buf)) \
    {                                              \
        NCNN_LOGE("read " #buf " failed");         \
        return -1;                                 \
    }
#endif
    int magic = 0;
    READ_VALUE(magic)
    if (magic != 7767517)
    {
        NCNN_LOGE("param is too old, please regenerate");
        return -1;
    }

    int layer_count = 0;
    int blob_count = 0;
    READ_VALUE(layer_count)
    READ_VALUE(blob_count)
    if (layer_count <= 0 || blob_count <= 0)
    {
        NCNN_LOGE("invalid layer_count or blob_count");
        return -1;
    }

    layers.resize(layer_count);
    blobs.resize(blob_count);

    ParamDict pd;

    for (int i=0; i<layer_count; i++)
    {
        int typeindex;
        int bottom_count;
        int top_count;
        READ_VALUE(typeindex)
        READ_VALUE(bottom_count)
        READ_VALUE(top_count)

        Layer* layer = create_layer(typeindex);
        if (!layer)
        {
            int custom_index = typeindex & ~LayerType::CustomBit;
            layer = create_custom_layer(custom_index);
        }
        if (!layer)
        {
            NCNN_LOGE("layer %d not exists or registered", typeindex);
            clear();
            return -1;
        }

//         layer->type = std::string(layer_type);
//         layer->name = std::string(layer_name);
//         NCNN_LOGE("new layer %d", typeindex);

        layer->bottoms.resize(bottom_count);
        for (int j=0; j<bottom_count; j++)
        {
            int bottom_blob_index;
            READ_VALUE(bottom_blob_index)

            Blob& blob = blobs[bottom_blob_index];

            blob.consumers.push_back(i);

            layer->bottoms[j] = bottom_blob_index;
        }

        layer->tops.resize(top_count);
        for (int j=0; j<top_count; j++)
        {
            int top_blob_index;
            READ_VALUE(top_blob_index)

            Blob& blob = blobs[top_blob_index];

//             blob.name = std::string(blob_name);
//             NCNN_LOGE("new blob %s", blob_name);
            blob.producer = i;

            layer->tops[j] = top_blob_index;
        }

        // layer specific params
        int pdlr = pd.load_param_bin(dr);
        if (pdlr != 0)
        {
            NCNN_LOGE("ParamDict load_param_bin %d failed", i);
            continue;
        }

        int lr = layer->load_param(pd);
        if (lr != 0)
        {
            NCNN_LOGE("layer load_param failed");
            continue;
        }

        layers[i] = layer;
    }

#undef READ_VALUE
    return 0;
}

int Net::load_model(const DataReader& dr)
{
    if (layers.empty())
    {
        NCNN_LOGE("network graph not ready");
        return -1;
    }

    // load file
    int ret = 0;

    ModelBinFromDataReader mb(dr);
    for (size_t i=0; i<layers.size(); i++)
    {
        Layer* layer = layers[i];

        int lret = layer->load_model(mb);
        if (lret != 0)
        {
            NCNN_LOGE("layer load_model %d failed", (int)i);
            ret = -1;
            break;
        }
    }

    return ret;
}

#if NCNN_STDIO
#if NCNN_STRING
int Net::load_param(FILE* fp)
{
    DataReaderFromStdio dr(fp);
    return load_param(dr);
}

int Net::load_param(const char* protopath)
{
    FILE* fp = fopen(protopath, "rb");
    if (!fp)
    {
        NCNN_LOGE("fopen %s failed", protopath);
        return -1;
    }

    int ret = load_param(fp);
    fclose(fp);
    return ret;
}
#endif // NCNN_STRING

int Net::load_param_bin(FILE* fp)
{
    DataReaderFromStdio dr(fp);
    return load_param_bin(dr);
}

int Net::load_param_bin(const char* protopath)
{
    FILE* fp = fopen(protopath, "rb");
    if (!fp)
    {
        NCNN_LOGE("fopen %s failed", protopath);
        return -1;
    }

    int ret = load_param_bin(fp);
    fclose(fp);
    return ret;
}

int Net::load_model(FILE* fp)
{
    DataReaderFromStdio dr(fp);
    return load_model(dr);
}

int Net::load_model(const char* modelpath)
{
    FILE* fp = fopen(modelpath, "rb");
    if (!fp)
    {
        NCNN_LOGE("fopen %s failed", modelpath);
        return -1;
    }

    int ret = load_model(fp);
    fclose(fp);
    return ret;
}
#endif // NCNN_STDIO

int Net::load_param(const unsigned char* _mem)
{
    const unsigned char* mem = _mem;
    DataReaderFromMemory dr(mem);
    load_param_bin(dr);
    return static_cast<int>(mem - _mem);
}

int Net::load_model(const unsigned char* _mem)
{
    const unsigned char* mem = _mem;
    DataReaderFromMemory dr(mem);
    load_model(dr);
    return static_cast<int>(mem - _mem);
}

void Net::clear()
{
    blobs.clear();
    for (size_t i=0; i<layers.size(); i++)
    {
        delete layers[i];
    }
    layers.clear();
}

Extractor Net::create_extractor() const
{
    return Extractor(this, blobs.size());
}

#if NCNN_STRING
int Net::find_blob_index_by_name(const char* name) const
{
    for (size_t i=0; i<blobs.size(); i++)
    {
        const Blob& blob = blobs[i];
        if (blob.name == name)
        {
            return i;
        }
    }

    NCNN_LOGE("find_blob_index_by_name %s failed", name);
    return -1;
}

int Net::find_layer_index_by_name(const char* name) const
{
    for (size_t i=0; i<layers.size(); i++)
    {
        const Layer* layer = layers[i];
        if (layer->name == name)
        {
            return i;
        }
    }

    NCNN_LOGE("find_layer_index_by_name %s failed", name);
    return -1;
}

int Net::custom_layer_to_index(const char* type)
{
    const int custom_layer_registry_entry_count = custom_layer_registry.size();
    for (int i=0; i<custom_layer_registry_entry_count; i++)
    {
        if (strcmp(type, custom_layer_registry[i].name) == 0)
            return i;
    }

    return -1;
}

Layer* Net::create_custom_layer(const char* type)
{
    int index = custom_layer_to_index(type);
    if (index == -1)
        return 0;

    return create_custom_layer(index);
}
#endif // NCNN_STRING

Layer* Net::create_custom_layer(int index)
{
    const int custom_layer_registry_entry_count = custom_layer_registry.size();
    if (index < 0 || index >= custom_layer_registry_entry_count)
        return 0;

    layer_creator_func layer_creator = custom_layer_registry[index].creator;
    if (!layer_creator)
        return 0;

    return layer_creator();
}

int Net::forward_layer(int layer_index, std::vector<Mat>& blob_mats, bool lightmode) const
{
    const Layer* layer = layers[layer_index];

//     NCNN_LOGE("forward_layer %d %s", layer_index, layer->name.c_str());

    if (layer->one_blob_only)
    {
        // load bottom blob
        int bottom_blob_index = layer->bottoms[0];
        int top_blob_index = layer->tops[0];

        if (blob_mats[bottom_blob_index].dims == 0)
        {
            int ret = forward_layer(blobs[bottom_blob_index].producer, blob_mats, lightmode);
            if (ret != 0)
                return ret;
        }

        Mat bottom_blob = blob_mats[bottom_blob_index];

        if (lightmode)
        {
            // delete after taken in light mode
            blob_mats[bottom_blob_index].release();
            // deep copy for inplace forward if data is shared
            if (layer->support_inplace && *bottom_blob.refcount != 1)
            {
                bottom_blob = bottom_blob.clone();
            }
        }

        // forward
        if (lightmode && layer->support_inplace)
        {
            Mat& bottom_top_blob = bottom_blob;
#if NCNN_BENCHMARK
            double start = get_current_time();
            int ret = layer->forward_inplace(bottom_top_blob);
            double end = get_current_time();
            benchmark(layer, bottom_top_blob, bottom_top_blob, start, end);
#else
            int ret = layer->forward_inplace(bottom_top_blob);
#endif // NCNN_BENCHMARK
            if (ret != 0)
                return ret;

            // store top blob
            blob_mats[top_blob_index] = bottom_top_blob;
        }
        else
        {
            Mat top_blob;
#if NCNN_BENCHMARK
            double start = get_current_time();
            int ret = layer->forward(bottom_blob, top_blob);
            double end = get_current_time();
            benchmark(layer, bottom_blob, top_blob, start, end);
#else
            int ret = layer->forward(bottom_blob, top_blob);
#endif // NCNN_BENCHMARK
            if (ret != 0)
                return ret;

            // store top blob
            blob_mats[top_blob_index] = top_blob;
        }

    }
    else
    {
        // load bottom blobs
        std::vector<Mat> bottom_blobs;
        bottom_blobs.resize(layer->bottoms.size());
        for (size_t i=0; i<layer->bottoms.size(); i++)
        {
            int bottom_blob_index = layer->bottoms[i];

            if (blob_mats[bottom_blob_index].dims == 0)
            {
                int ret = forward_layer(blobs[bottom_blob_index].producer, blob_mats, lightmode);
                if (ret != 0)
                    return ret;
            }

            bottom_blobs[i] = blob_mats[bottom_blob_index];

            if (lightmode)
            {
                // delete after taken in light mode
                blob_mats[bottom_blob_index].release();
                // deep copy for inplace forward if data is shared
                if (layer->support_inplace && *bottom_blobs[i].refcount != 1)
                {
                    bottom_blobs[i] = bottom_blobs[i].clone();
                }
            }
        }

        // forward
        if (lightmode && layer->support_inplace)
        {
            std::vector<Mat>& bottom_top_blobs = bottom_blobs;
#if NCNN_BENCHMARK
            double start = get_current_time();
            int ret = layer->forward_inplace(bottom_top_blobs);
            double end = get_current_time();
            benchmark(layer, start, end);
#else
            int ret = layer->forward_inplace(bottom_top_blobs);
#endif // NCNN_BENCHMARK
            if (ret != 0)
                return ret;

            // store top blobs
            for (size_t i=0; i<layer->tops.size(); i++)
            {
                int top_blob_index = layer->tops[i];

                blob_mats[top_blob_index] = bottom_top_blobs[i];
            }
        }
        else
        {
            std::vector<Mat> top_blobs;
            top_blobs.resize(layer->tops.size());
#if NCNN_BENCHMARK
            double start = get_current_time();
            int ret = layer->forward(bottom_blobs, top_blobs);
            double end = get_current_time();
            benchmark(layer, start, end);
#else
            int ret = layer->forward(bottom_blobs, top_blobs);
#endif // NCNN_BENCHMARK
            if (ret != 0)
                return ret;

            // store top blobs
            for (size_t i=0; i<layer->tops.size(); i++)
            {
                int top_blob_index = layer->tops[i];

                blob_mats[top_blob_index] = top_blobs[i];
            }
        }
    }

//     NCNN_LOGE("forward_layer %d %s done", layer_index, layer->name.c_str());
//     const Mat& blob = blob_mats[layer->tops[0]];
//     NCNN_LOGE("[%-2d %-16s %-16s]  %d    blobs count = %-3d   size = %-3d x %-3d", layer_index, layer->type.c_str(), layer->name.c_str(), layer->tops[0], blob.c, blob.h, blob.w);

    return 0;
}

class ExtractorPrivate
{
public:
    ExtractorPrivate(const Net* _net)
        : net(_net)
    {
    }
    const Net* net;
    std::vector<Mat> blob_mats;
    bool lightmode;
    int num_threads;
};

Extractor::Extractor(const Net* _net, int blob_count)
    : d(new ExtractorPrivate(_net))
{
    d->blob_mats.resize(blob_count);
    lightmode = true;
    num_threads = 0;
}

void Extractor::set_light_mode(bool enable)
{
    lightmode = enable;
}

void Extractor::set_num_threads(int _num_threads)
{
    num_threads = _num_threads;
}

int Extractor::input(int blob_index, const Mat& in)
{
    if (blob_index < 0 || blob_index >= (int)d->blob_mats.size())
        return -1;

    d->blob_mats[blob_index] = in;

    return 0;
}

int Extractor::extract(int blob_index, Mat& feat)
{
    if (blob_index < 0 || blob_index >= (int)d->blob_mats.size())
        return -1;

    int ret = 0;

    if (d->blob_mats[blob_index].dims == 0)
    {
        int layer_index = d->net->blobs[blob_index].producer;

#ifdef _OPENMP
        int dynamic_current = 0;
        int num_threads_current = 1;
        if (num_threads)
        {
            dynamic_current = omp_get_dynamic();
            num_threads_current = omp_get_num_threads();
            omp_set_dynamic(0);
            omp_set_num_threads(num_threads);
        }
#endif

        ret = d->net->forward_layer(layer_index, d->blob_mats, lightmode);

#ifdef _OPENMP
        if (num_threads)
        {
            omp_set_dynamic(dynamic_current);
            omp_set_num_threads(num_threads_current);
        }
#endif
    }

    feat = d->blob_mats[blob_index];

    return ret;
}

#if NCNN_STRING
int Extractor::input(const char* blob_name, const Mat& in)
{
    int blob_index = d->net->find_blob_index_by_name(blob_name);
    if (blob_index == -1)
        return -1;

    d->blob_mats[blob_index] = in;

    return 0;
}

int Extractor::extract(const char* blob_name, Mat& feat)
{
    int blob_index = d->net->find_blob_index_by_name(blob_name);
    if (blob_index == -1)
        return -1;

    int ret = 0;

    if (d->blob_mats[blob_index].dims == 0)
    {
        int layer_index = d->net->blobs[blob_index].producer;

#ifdef _OPENMP
        int dynamic_current = 0;
        int num_threads_current = 1;
        if (num_threads)
        {
            dynamic_current = omp_get_dynamic();
            num_threads_current = omp_get_num_threads();
            omp_set_dynamic(0);
            omp_set_num_threads(num_threads);
        }
#endif

        ret = d->net->forward_layer(layer_index, d->blob_mats, lightmode);

#ifdef _OPENMP
        if (num_threads)
        {
            omp_set_dynamic(dynamic_current);
            omp_set_num_threads(num_threads_current);
        }
#endif
    }

    feat = d->blob_mats[blob_index];

    return ret;
}
#endif // NCNN_STRING

} // namespace ncnn

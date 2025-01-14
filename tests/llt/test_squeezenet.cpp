#include <gtest/gtest.h>
#include "testutil.h"
#include <stdio.h>
#include <algorithm>
#include <vector>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "net.h"

static void detect_squeezenet(const cv::Mat &bgr, std::vector<float> &cls_scores)
{
    ncnn::Net squeezenet;
    squeezenet.opt.use_vulkan_compute = false;
    squeezenet.load_param("../../demo/model/squeezenet_v1.1.param");
    squeezenet.load_model("../../demo/model/squeezenet_v1.1.bin");

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, 227, 227);

    const float mean_vals[3] = {104.f, 117.f, 123.f};
    in.substract_mean_normalize(mean_vals, 0);

    ncnn::Extractor ex = squeezenet.create_extractor();
    ex.set_light_mode(true);

    ex.input("data", in);

    ncnn::Mat out;
    ex.extract("prob", out);

    cls_scores.resize(out.w);
    for (int j = 0; j < out.w; j++)
    {
        cls_scores[j] = out[j];
    }
}

static void prob_sort(const std::vector<float> &cls_scores, int topk, int *ret)
{
    // partial sort topk with index
    int size = cls_scores.size();
    std::vector<std::pair<float, int>> vec;
    vec.resize(size);
    for (int i = 0; i < size; i++)
    {
        vec[i] = std::make_pair(cls_scores[i], i);
    }

    std::partial_sort(vec.begin(), vec.begin() + topk, vec.end(),
                      std::greater<std::pair<float, int>>());

    // print topk and score
    for (int i = 0; i < topk; i++)
    {
        float score = vec[i].first;
        int index = vec[i].second;
        fprintf(stderr, "%d = %f\n", index, score);
        ret[i] = index;
    }
}

TEST(squeezenetTest, UTGeneral)
{
    std::string imagePath = "../../demo/images/cat.jpg";
    // const char* imagepath = argv[1];
    cv::Mat m = cv::imread(imagePath, cv::IMREAD_COLOR);
    ASSERT_TRUE(!m.empty());

    std::vector<float> cls_scores;
    detect_squeezenet(m, cls_scores);

    int top_k = 3;
    int ans[top_k];
    prob_sort(cls_scores, top_k, ans);
    int expect_ans[] = {283, 279, 287};
    for (int i = 0; i < top_k; ++i)
    {
        EXPECT_EQ(ans[i], expect_ans[i]);
    }
}
#pragma once

void StereoEstimation_Naive(
    const int &window_size,
    const int &dmin,
    cv::Mat &image1, cv::Mat &image2,
    cv::Mat &naive_disparities, const double &scale);

void DynamicApproach_calculation(
    const double &lambda,
    cv::Mat &image1, cv::Mat &image2,
    cv::Mat &dp_disparities, const double &scale);

void Disparity2PointCloud(
    const std::string &output_file,
    int height, int width, cv::Mat &disparities,
    const int &window_size,
    const int &dmin, const double &baseline, const double &focal_length);

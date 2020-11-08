#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <fstream>
#include "main.h"


int main(int argc, char** argv) {

    ////////////////
    // Parameters //
    ////////////////

    // camera setup parameters
    const double focal_length = 1247;
    const double baseline = 213;

    // stereo estimation parameters


    const double scale = 3;
    int dmin = 67;
    if (argc > 4) dmin = std::stoi(argv[4]);
    int window_size = 5;
    if (argc > 5) window_size = std::stoi(argv[5]);
    double weight = 500;
    if (argc > 6) weight = std::stod(argv[6]);

    ///////////////////////////
    // Commandline arguments //
    ///////////////////////////

    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " IMAGE1 IMAGE2 OUTPUT_FILE [dmin] [window_size] [weight] [scale]"
            << std::endl;
        return 1;
    }

    cv::Mat image1 = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    cv::Mat image2 = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);
    const std::string output_file = argv[3];

    if (!image1.data) {
        std::cerr << "No image1 data" << std::endl;
        return EXIT_FAILURE;
    }

    if (!image2.data) {
        std::cerr << "No image2 data" << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "------------------ Parameters -------------------" << std::endl;
    std::cout << "focal_length = " << focal_length << std::endl;
    std::cout << "baseline = " << baseline << std::endl;
    std::cout << "window_size = " << window_size << std::endl;
    std::cout << "occlusion weights = " << weight << std::endl;
    std::cout << "disparity added due to image cropping = " << dmin << std::endl;
    std::cout << "scaling of disparity images to show = " << scale << std::endl;
    std::cout << "output filename = " << argv[3] << std::endl;
    std::cout << "-------------------------------------------------" << std::endl;

    int height = image1.size().height;
    int width = image1.size().width;

    ////////////////////
    // Reconstruction //
    ////////////////////

    // Naive disparity image
    //cv::Mat naive_disparities = cv::Mat::zeros(height - window_size, width - window_size, CV_8UC1);
    cv::Mat naive_disparities = cv::Mat::zeros(height, width, CV_8UC1);
    cv::Mat dynamic_disparities = cv::Mat::zeros(height, width, CV_8UC1);



    StereoEstimation_Naive(
        window_size, dmin, image1, image2,
        naive_disparities, scale);
    DynamicApproach_calculation(
        weight, image1, image2,
        dynamic_disparities, scale);

    ////////////
    // Output //
    ////////////

    // reconstruction
    Disparity2PointCloud(
        output_file + "_Naive_Approach",
        height, width, naive_disparities,
        window_size, dmin, baseline, focal_length);

    // save / display images
    std::stringstream out1;
    out1 << output_file << "_Naive_Approach.png";
    cv::imwrite(out1.str(), naive_disparities);

    cv::namedWindow("Naive stereo matching", cv::WINDOW_AUTOSIZE);
    cv::imshow("Naive stereo matching", naive_disparities);

    // reconstruction
    Disparity2PointCloud(
        output_file + "_Dynamic_Approach",
        height, width, dynamic_disparities,
        window_size, dmin, baseline, focal_length);

    // save / display images
    std::stringstream out2;
    out2 << output_file << "_Dynamic_Approach.png";
    cv::imwrite(out2.str(), dynamic_disparities);

    cv::namedWindow("Dynamic programming approach", cv::WINDOW_AUTOSIZE);
    cv::imshow("Dynamic programming approach", dynamic_disparities);

    cv::waitKey(0);

    return 0;
}

void StereoEstimation_Naive(
    const int& window_size,
    const int& dmin,
    cv::Mat& image1, cv::Mat& image2,
    cv::Mat& naive_disparities, const double& scale) {
    int height = image1.size().height; //ad
    int width = image1.size().width; //ad
    int half_window_size = window_size / 2;

    for (int i = half_window_size; i < height - half_window_size; ++i) {

        std::cout
            << "Calculating disparities for the naive approach... "
            << std::ceil(((i - half_window_size + 1) / static_cast<double>(height - window_size + 1)) * 100) << "%\r"
            << std::flush;

        for (int j = half_window_size; j < width - half_window_size; ++j) {
            int min_ssd = INT_MAX;
            int disparity = 0;

            for (int d = -j + half_window_size; d < width - j - half_window_size; ++d) {
                int ssd = 0;
                // TODO: sum up matching cost (ssd) in a window
                for (int m = -half_window_size; m <= half_window_size; ++m) {
                    for (int n = -half_window_size; n <= half_window_size; ++n) {

                        int v_left = image1.at<uchar>(i + m, j + n);
                        int v_right = image2.at<uchar>(i + m, j + n + d);
                        int diff = v_left - v_right;

                            ssd += diff * diff;
                    }
                }
                //Sum of Squared Differences
                if (ssd < min_ssd) {
                    min_ssd = ssd;
                    disparity = d;
                }
            }

            naive_disparities.at<uchar>(i - half_window_size, j - half_window_size) = std::abs(disparity) * scale;
        }
    }

    std::cout << "Calculating disparities for the naive approach... Done.\r" << std::flush;
    std::cout << std::endl;
}

void DynamicApproach_calculation(
    const double& lambda,
    cv::Mat& image1, cv::Mat& image2, cv::Mat& dynamic_disparities, const double& scale) {

    //Store the table of costs C
    //Store preceding nodes in M
    cv::Mat M;
    cv::Mat C;
    //rs-rows,r=row,c-columns
    int rows = image1.rows; int cols = image1.cols;
    for (int row = 0; row < rows; ++row) {

        std::cout
            << "Calculating disparities for the Dynamic programming  approach... "
            << std::ceil(((row + 1) / static_cast<double>(rows + 1)) * 100) << "%\r"
            << std::flush;
        M = cv::Mat::zeros(cols, cols, CV_8UC1); C = cv::Mat::zeros(cols, cols, CV_64FC1);

        for (int i = 0; i < cols; i++) {
            C.at<double>(i, 0) = i * lambda;
            C.at<double>(0, i) = i * lambda;
        }

        for (int i = 1; i < cols; ++i) {
            for (int j = 1; j < cols; ++j) {

                uchar dissim = image1.at<uchar>(row, i) - image2.at<uchar>(row, j);
                double match = C.at<double>(i - 1, j - 1) + dissim * dissim; // match
                double lftOccl = C.at<double>(i - 1, j) + lambda; // occluded from left

                double rghtoccl = C.at<double>(i, j - 1) + lambda; // occluded from right

                //Populate C and M matrices depending on the minimum of the computed values

                double min = std::min(std::min(match, lftOccl), rghtoccl);
                if (min == match) M.at<uchar>(i, j) = 1;
                else if (min == lftOccl) M.at<uchar>(i, j) = 2;
                else if (min == rghtoccl) M.at<uchar>(i, j) = 3;

                C.at<double>(i, j) = min;
            }
        }

        int m = cols - 1, n = cols - 1;
        while (m > 0 && n > 0) {
            int Occlusionvalue = M.at<uchar>(m, n);
            switch (Occlusionvalue) {
                // match
            case 1:
                m--;
                n--;
                break;
            case 2: //case Left Occlusion
                m--;
                break;
            case 3: //case Right Occlusion
                n--;
                break;
            }
            dynamic_disparities.at<uchar>(row, m) = abs(m - n) * scale;
        }
    }

    std::cout << "Calculating disparities for the DP approach... Done.\r" << std::flush;
    std::cout << std::endl;
}

void Disparity2PointCloud(
    const std::string& output_file,
    int height, int width, cv::Mat& disparities,
    const int& window_size,
    const int& dmin, const double& baseline, const double& focal_length) {
    std::stringstream out3d;
    out3d << output_file << ".xyz";
    std::ofstream outfile(out3d.str());
    for (int i = 0; i < height - window_size; ++i) {
        std::cout << "Reconstructing 3D point cloud from disparities... "
            << std::ceil(((i) / static_cast<double>(height - window_size + 1)) * 100) << "%\r" << std::flush;
        for (int j = 0; j < width - window_size; ++j) {
            if (disparities.at<uchar>(i, j) == 0) continue;

            const double W = baseline / (disparities.at<uchar>(i, j) + dmin);

            //traingulation
            const double Z = focal_length * baseline / (disparities.at<uchar>(i, j) + dmin);
            const double X = W*j / focal_length;
            const double Y = W*i/ focal_length;

            outfile << X << " " << Y << " " << Z << std::endl;
        }
    }

    std::cout << "Reconstructing 3D point cloud from disparities... Done.\r" << std::flush;
    std::cout << std::endl;
}

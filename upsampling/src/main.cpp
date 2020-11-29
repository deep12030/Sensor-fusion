#include <iostream>
#include <opencv2/opencv.hpp>
#include <math.h>
# define M_PI           3.141
using namespace std;
using namespace cv;

/*
Subtask 1: Implement bilateral filter
- Evaluate the filter on an image of your choice
- Choose four different levels of sigmas for the spatial and four for the spectral filter.
- Test all combinations, 16 in total, by running the Bilateral filter on the image
Subtask 2: Convert bilateral filter to guided joint bilateral filter for guided image upsampling
- apply filter to upsample the depth image, guided by an RGB image
*/


//working with metrices 

double SSD_Metrices(const Mat& origImg, const Mat& denoisedImg)
{
	double ssd = 0; double diff = 0;
	for (int r = 0; r < origImg.rows; ++r) {
		for (int c = 0; c < origImg.cols; ++c) {
			diff = origImg.at<uchar>(r, c) - denoisedImg.at<uchar>(r, c);
			ssd += diff * diff;
		}
	}
	return ssd;
}

double RMSE_Metrices(const cv::Mat& origImg, const cv::Mat& denoisedImg)
{
	int size = origImg.rows * origImg.cols;
	double ssd = 0; double diff = 0;
	for (int r = 0; r < origImg.rows; ++r) {
		for (int c = 0; c < origImg.cols; ++c) {
			diff = origImg.at<uchar>(r, c) - denoisedImg.at<uchar>(r, c);
			ssd += diff * diff;
		}
	}
	double mse = (double)(ssd / size);
	return sqrt(mse);
}

double PSNR_Metrices(const cv::Mat& origImg, const cv::Mat& denoisedImg)
{

	double max = 255; double ssd = 0; double diff = 0;
	int size = origImg.rows * origImg.cols;
	for (int r = 0; r < origImg.rows; ++r) {
		for (int c = 0; c < origImg.cols; ++c) {
			diff = origImg.at<uchar>(r, c) - denoisedImg.at<uchar>(r, c);
			ssd += diff * diff;
		}
	}
	double mse = (double)(ssd / size);
	double psnr = 10 * log10((max * max) / mse);
	return psnr;
}


//working with filter

cv::Mat CreateGaussianKernel_orig(int window_size) {
	cv::Mat kernel(cv::Size(window_size, window_size), CV_32FC1);

	int half_window_size = window_size / 2;

	// see: lecture_03_slides.pdf, Slide 13
	const double k = 2.5;
	const double r_max = std::sqrt(2.0 * half_window_size * half_window_size);
	const double sigma = r_max / k;

	// sum is for normalization 
	float sum = 0.0;

	for (int x = -window_size / 2; x <= window_size / 2; x++) {
		for (int y = -window_size / 2; y <= window_size / 2; y++) {
			float val = exp(-(x * x + y * y) / (2 * sigma * sigma));
			kernel.at<float>(x + window_size / 2, y + window_size / 2) = val;
			sum += val;
		}
	}

	// normalising the Kernel 
	for (int i = 0; i < 5; ++i)
		for (int j = 0; j < 5; ++j)
			kernel.at<float>(i, j) /= sum;

	// note that this is a naive implementation
	// there are alternative (better) ways
	// e.g. 
	// - perform analytic normalisation (what's the integral of the gaussian? :))
	// - you could store and compute values as uchar directly in stead of float
	// - computing it as a separable kernel [ exp(x + y) = exp(x) * exp(y) ] ...
	// - ...

	return kernel;
}

cv::Mat GaussianKernelFunction(int window_size, float sigma = 1) // 0.1 ... 3
{
	cv::Mat kernel(window_size, window_size, CV_32FC1);
	double sum = 0.0; double i, j;
	for (i = 0; i < window_size; i++) {
		for (j = 0; j < window_size; j++) {
			kernel.at<float>(i, j) = exp(-(i * i + j * j) / (2 * sigma * sigma)) / (2 * M_PI * sigma * sigma);
			sum += kernel.at<float>(i, j);
		}
	}
	for (i = 0; i < window_size; i++) {
		for (j = 0; j < window_size; j++) {
			kernel.at<float>(i, j) /= sum;
		}
	}
	return kernel;
}

cv::Mat GaussianFilterFunction(const cv::Mat& input) {
	cv::Mat output(input.size(), input.type());
	const auto wdth = input.cols; const auto hght = input.rows; const int window_size = 5;
	cv::Mat gaussianKernel = GaussianKernelFunction(window_size);
	for (int r = 0; r < hght; ++r) {
		for (int c = 0; c < wdth; ++c) {
			output.at<uchar>(r, c) = 0;
		}
	}

	for (int r = window_size / 2; r < hght - window_size / 2; ++r) {
		for (int c = window_size / 2; c < wdth - window_size / 2; ++c) {

			int sum = 0;
			for (int i = -window_size / 2; i <= window_size / 2; ++i) {
				for (int j = -window_size / 2; j <= window_size / 2; ++j) {
					sum += input.at<uchar>(r + i, c + j) * gaussianKernel.at<float>(i + window_size / 2, j + window_size / 2);
				}
			}
			output.at<uchar>(r, c) = sum;
		}
	}
	return output;
}

cv::Mat OurFilter_Bilateral(const cv::Mat& input, const int window_size = 5, float spatial_sigma = 1, float spectral_sigma = 5) {
	const auto wdth = input.cols; const auto hght = input.rows;
	cv::Mat output(input.size(), input.type());
	// sigma spatial filter (Gaussian, \(w_G\) kernel)
	cv::Mat gaussianKernel = GaussianKernelFunction(window_size, spatial_sigma);

	for (int r = 0; r < hght; ++r) {
		for (int c = 0; c < wdth; ++c) {
			output.at<uchar>(r, c) = 0;
		}
	}

	auto d = [](float a, float b) {
		return std::abs(a - b);
	};

	auto p = [](float val, float sigma) {
		const float sigmaSq = sigma * sigma;
		const float normalization = std::sqrt(2 * M_PI) * sigma;
		return (1 / normalization) * std::exp(-val / (2 * sigmaSq));
	};

	for (int r = window_size / 2; r < hght - window_size / 2; ++r) {
		for (int c = window_size / 2; c < wdth - window_size / 2; ++c) {

			float sum_w = 0;
			float sum = 0;

			for (int i = -window_size / 2; i <= window_size / 2; ++i) {
				for (int j = -window_size / 2; j <= window_size / 2; ++j) {

					float range_diff
						= d(input.at<uchar>(r, c), input.at<uchar>(r + i, c + j));

					float w
						= p(range_diff, spectral_sigma) // spectral 
						* gaussianKernel.at<float>(i + window_size / 2, j + window_size / 2); // spatial 

					sum
						+= input.at<uchar>(r + i, c + j) * w;
					sum_w
						+= w;
				}
			}

			output.at<uchar>(r, c) = sum / sum_w;

		}
	}
	return output;
}

void JointBilateralFilter(const cv::Mat& input_rgb, const cv::Mat& input_depth, cv::Mat& output, const int window_size = 5, float sigma = 5) {
	const auto wdth = input_rgb.cols; const auto hght = input_rgb.rows;

	cv::Mat gaussianKernel = GaussianKernelFunction(window_size, 0.5);
	auto d = [](float a, float b) {
		return std::abs(a - b);
	};

	auto p = [](float val, float sigma) {	// use of weighting function 
		const float sigmaSq = sigma * sigma;
		const float normalization = std::sqrt(2 * M_PI) * sigma;
		return (1 / normalization) * std::exp(-val / (2 * sigmaSq));
	};

	for (int r = window_size / 2; r < hght - window_size / 2; ++r) {
		for (int c = window_size / 2; c < wdth - window_size / 2; ++c) {

			float sum_w = 0; float sum = 0;

			for (int i = -window_size / 2; i <= window_size / 2; ++i) {
				for (int j = -window_size / 2; j <= window_size / 2; ++j) {

					float range_diff
						= d(input_rgb.at<uchar>(r, c), input_rgb.at<uchar>(r + i, c + j));

					float w
						= p(range_diff, sigma)
						* gaussianKernel.at<float>(i + window_size / 2, j + window_size / 2);

					sum
						+= input_depth.at<uchar>(r + i, c + j) * w;
					sum_w
						+= w;
				}
			}

			output.at<uchar>(r, c) = sum / sum_w;

		}
	}
}
//slides
cv::Mat Upsampling(const cv::Mat& input_rgb, const cv::Mat& input_depth) {
	//  iterative upsampling
	int uf = log2(input_rgb.rows / input_depth.rows); // upsample factor
	// low resolution depth image
	cv::Mat D = input_depth.clone();
	// high resolution color image
	cv::Mat I = input_rgb.clone();
	for (int i = 0; i < uf; ++i)
	{
		cv::resize(D, D, D.size() * 2);
		cv::resize(I, I, D.size());
		JointBilateralFilter(I, D, D, 5, 0.1);
	}
	cv::resize(D, D, input_rgb.size());
	JointBilateralFilter(input_rgb, D, D, 5, 0.1);
	return D;
}



int main(int argc, char** argv) {

	if (argc < 2) {
		std::cerr << "Usage: " << argv[0] << " DATA_PATH" << std::endl;
		return 1;
	}

	std::string path = "C:\\Users\\kumar\\Downloads\\Bilateral SF\\input\\";
	cv::Mat rgbImageInput = cv::imread(path + "im2.png", 0);
	cv::Mat InputDispDepth = cv::imread(path + "LR_disp2.png", 0);
	cv::Mat image = cv::imread(path + "lena.png", 0);

	if (image.data == nullptr) {
		std::cerr << "image load failed" << std::endl;
	}
	cv::Mat input = image.clone();
	cv::Mat noise(image.size(), image.type());
	uchar mean = 0; uchar standard_dev = 25;
	cv::randn(noise, mean, standard_dev);
	image += noise;

	std::string outPath = "C:\\Users\\kumar\\Downloads\\Bilateral SF\\out\\";

	cv::Mat upSampled = Upsampling(rgbImageInput, InputDispDepth);
	// converting the bilateral filter to Guided Joint bilateral filter for guided image upsampling
	imwrite(outPath + "upsampled Depth Image.PNG", upSampled);
	std::vector<float> spat_sgma = { 1, 2.5, 3, 4.5 };
	std::vector<float> spctrl_sgma = { 1, 5, 7, 9 };
	int count = 0;
	for (auto i : spat_sgma) {	// evaluating the filter on an image 
		for (auto j : spctrl_sgma) {
			count = count + 1;
			cv::Mat outputBilateral = OurFilter_Bilateral(image, 5, i, j); // implementation of the bilateral filter
			double ssd = SSD_Metrices(input, outputBilateral);
			double rmse = RMSE_Metrices(input, outputBilateral);
			double psnr = PSNR_Metrices(input, outputBilateral);
			std::cout << "|Combination=>|  " << std::to_string(count) << " |spat_sgma| " << std::to_string(i) << "|spctrl_sgma|" << std::to_string(j) << std::endl;
			std::cout << "psnr = " << std::to_string(psnr) << " ";
			std::cout << "ssd = " << std::to_string(ssd) << " ";
			//std::cout << "perform = " << PerformanceMetrics1 << " ";
			std::cout << "rmse = " << std::to_string(rmse) << std::endl;
			imwrite(outPath + std::to_string(count) + " " + "spatial" + std::to_string(i) + "spectral"  // saving the result
				+ std::to_string(j) + ".PNG", outputBilateral);
		}
	}
	std::cin.get();
	return 0;

}


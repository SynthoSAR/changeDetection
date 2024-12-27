#include <cuda_runtime.h>
#include <iostream>
#include <opencv2/opencv.hpp>

#define HISTOGRAM_BINS 256
#define BLOCK_SIZE 16

// CUDA kernel to compute local histograms
__global__ void computeLocalHistogram(const unsigned char* image, int* histograms, int imageWidth, int imageHeight, int blockWidth, int blockHeight) {
    int blockX = blockIdx.x * blockDim.x + threadIdx.x;
    int blockY = blockIdx.y * blockDim.y + threadIdx.y;

    if (blockX < blockWidth && blockY < blockHeight) {
        int blockIdx = blockY * blockWidth + blockX;
        int startX = blockX * BLOCK_SIZE;
        int startY = blockY * BLOCK_SIZE;

        // Compute local histogram
        for (int y = 0; y < BLOCK_SIZE; ++y) {
            for (int x = 0; x < BLOCK_SIZE; ++x) {
                int imgX = startX + x;
                int imgY = startY + y;
                if (imgX < imageWidth && imgY < imageHeight) {
                    int pixelVal = image[imgY * imageWidth + imgX];
                    atomicAdd(&histograms[blockIdx * HISTOGRAM_BINS + pixelVal], 1);
                }
            }
        }
    }
}

// CUDA kernel to compute histogram differences
__global__ void computeLocalHistogramDifference(const int* hist1, const int* hist2, float* differences, int totalBlocks, int totalBins) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < totalBlocks * totalBins) {
        int blockIdx = idx / totalBins;
        int binIdx = idx % totalBins;

        differences[blockIdx * totalBins + binIdx] = abs(hist1[blockIdx * totalBins + binIdx] - hist2[blockIdx * totalBins + binIdx]);
    }
}

// CUDA kernel to compute pixel-level change map
__global__ void computeChangeMap(const unsigned char* image1, const unsigned char* image2, unsigned char* changeMap, int imageWidth, int imageHeight, float threshold) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < imageWidth * imageHeight) {
        int diff = abs(image1[idx] - image2[idx]);
        changeMap[idx] = (diff > threshold) ? 255 : 0;
    }
}

// Host function
void histogramBasedChangeDetection(cv::Mat& img1, cv::Mat& img2, float threshold) {
    int imageWidth = img1.cols;
    int imageHeight = img1.rows;
    int imageSize = imageWidth * imageHeight;
    int blockWidth = (imageWidth + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int blockHeight = (imageHeight + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int totalBlocks = blockWidth * blockHeight;
    int histogramSize = totalBlocks * HISTOGRAM_BINS * sizeof(int);
    int differenceSize = totalBlocks * HISTOGRAM_BINS * sizeof(float);

    // Host memory allocation
    int* hist1 = new int[totalBlocks * HISTOGRAM_BINS]();
    int* hist2 = new int[totalBlocks * HISTOGRAM_BINS]();
    float* differences = new float[totalBlocks * HISTOGRAM_BINS]();
    unsigned char* changeMap = new unsigned char[imageSize];

    // Device memory allocation
    unsigned char* d_image1, * d_image2, * d_changeMap;
    int* d_hist1, * d_hist2;
    float* d_differences;

    cudaMalloc(&d_image1, imageSize);
    cudaMalloc(&d_image2, imageSize);
    cudaMalloc(&d_changeMap, imageSize);
    cudaMalloc(&d_hist1, histogramSize);
    cudaMalloc(&d_hist2, histogramSize);
    cudaMalloc(&d_differences, differenceSize);

    // Copy images to device
    cudaMemcpy(d_image1, img1.data, imageSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_image2, img2.data, imageSize, cudaMemcpyHostToDevice);

    // Initialize histograms
    cudaMemset(d_hist1, 0, histogramSize);
    cudaMemset(d_hist2, 0, histogramSize);

    // Define grid and block dimensions
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid(blockWidth, blockHeight);

    // Launch histogram computation kernels
    computeLocalHistogram << <blocksPerGrid, threadsPerBlock >> > (d_image1, d_hist1, imageWidth, imageHeight, blockWidth, blockHeight);
    computeLocalHistogram << <blocksPerGrid, threadsPerBlock >> > (d_image2, d_hist2, imageWidth, imageHeight, blockWidth, blockHeight);

    // Compute histogram differences
    int threads = 256;
    int blocks = (totalBlocks * HISTOGRAM_BINS + threads - 1) / threads;
    computeLocalHistogramDifference << <blocks, threads >> > (d_hist1, d_hist2, d_differences, totalBlocks, HISTOGRAM_BINS);

    // Compute change map
    blocks = (imageSize + threads - 1) / threads;
    computeChangeMap << <blocks, threads >> > (d_image1, d_image2, d_changeMap, imageWidth, imageHeight, threshold);

    // Copy change map back to host
    cudaMemcpy(changeMap, d_changeMap, imageSize, cudaMemcpyDeviceToHost);

    // Visualize change map
    cv::Mat changeMapImage(imageHeight, imageWidth, CV_8UC1, changeMap);
    cv::imshow("Change Map", changeMapImage);
    cv::waitKey(0);

    // Free device memory
    cudaFree(d_image1);
    cudaFree(d_image2);
    cudaFree(d_changeMap);
    cudaFree(d_hist1);
    cudaFree(d_hist2);
    cudaFree(d_differences);

    // Free host memory
    delete[] hist1;
    delete[] hist2;
    delete[] differences;
    delete[] changeMap;
}

int main() {
    // Read input images using OpenCV
    cv::Mat img1 = cv::imread("bimg1.png", cv::IMREAD_GRAYSCALE);
    cv::Mat img2 = cv::imread("bimg2.png", cv::IMREAD_GRAYSCALE);

    if (img1.empty() || img2.empty()) {
        std::cerr << "Error: Could not open or find the images!" << std::endl;
        return -1;
    }

    // Ensure both images have the same size
    if (img1.size() != img2.size()) {
        std::cerr << "Error: Images must have the same dimensions!" << std::endl;
        return -1;
    }

    // Apply Gaussian filtering to reduce noise
    cv::GaussianBlur(img1, img1, cv::Size(5, 5), 0);
    cv::GaussianBlur(img2, img2, cv::Size(5, 5), 0);

    // Define a threshold for change detection
    float threshold = 30.0f;

    // Call the histogram-based change detection function
    histogramBasedChangeDetection(img1, img2, threshold);

    return 0;
}

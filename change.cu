#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <iostream>

using namespace cv;
using namespace cv::cuda;
using namespace std;

int main() {
    // Check if CUDA is available
    if (cuda::getCudaEnabledDeviceCount() == 0) {
        cout << "Error: No CUDA-capable device found." << endl;
        return -1;
    }

    // Load binary images directly to GPU
    Mat img1_host = imread("1.png", IMREAD_GRAYSCALE);
    Mat img2_host = imread("2.jpg", IMREAD_GRAYSCALE);

    if (img1_host.empty() || img2_host.empty()) {
        cout << "Error: Could not load images." << endl;
        return -1;
    }

    // Upload images to GPU
    GpuMat img1, img2;
    img1.upload(img1_host);
    img2.upload(img2_host);

    // Ensure images are binary (convert to 0 and 1)
    GpuMat img1_binary, img2_binary;
    cuda::threshold(img1, img1_binary, 128, 1, THRESH_BINARY);
    cuda::threshold(img2, img2_binary, 128, 1, THRESH_BINARY);

    // Step 1: Compute Difference Map (XOR operation)
    GpuMat diff;
    cuda::bitwise_xor(img1_binary, img2_binary, diff);

    // Step 2: Morphological Filtering to Remove Noise
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3)); // 3x3 kernel
    Ptr<cuda::Filter> morph_filter = cuda::createMorphologyFilter(MORPH_OPEN, diff.type(), kernel);
    GpuMat filtered;
    morph_filter->apply(diff, filtered);

    // Step 3: Exclude 30% from left and right, 10% from top and bottom
    int margin_x = static_cast<int>(img1.cols * 0.3); // 30% of width
    int margin_y = static_cast<int>(img1.rows * 0.1); // 10% of height
    Rect roi(margin_x, margin_y, img1.cols - 2 * margin_x, img1.rows - 2 * margin_y);

    // Create a mask on GPU for the central ROI
    GpuMat mask(filtered.size(), CV_8UC1, Scalar(0)); // Initialize to zeros
    GpuMat mask_roi = mask(roi);
    mask_roi.setTo(Scalar(255)); // Set central region to white (255)

    // Apply the mask to keep only the central area
    GpuMat filtered_roi;
    cuda::bitwise_and(filtered, mask, filtered_roi);

    // Step 4: Connected Component Analysis (CPU fallback due to limited CUDA support)
    Mat filtered_roi_host;
    filtered_roi.download(filtered_roi_host); // Transfer to CPU

    Mat labels, stats, centroids;
    int num_labels = connectedComponentsWithStats(filtered_roi_host, labels, stats, centroids);

    int min_size = 10; // Minimum size threshold
    int max_size = 20; // Maximum size threshold

    for (int i = 1; i < num_labels; i++) { // Skip background (label 0)
        int area = stats.at<int>(i, CC_STAT_AREA);
        if (area < min_size || area > max_size) {
            filtered_roi_host.setTo(0, labels == i);
        }
    }

    // Upload filtered result back to GPU
    filtered_roi.upload(filtered_roi_host);

    // Step 5: Find Contours to Highlight Changes (CPU-based due to OpenCV CUDA limitations)
    Mat filtered_roi_contours;
    filtered_roi.download(filtered_roi_contours);

    vector<vector<Point>> contours;
    findContours(filtered_roi_contours, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // Convert img1 to color for visualization
    Mat img1_color;
    Mat temp = img1_host * 255; // Resolve MatExpr to Mat
    cv::cvtColor(temp, img1_color, COLOR_GRAY2BGR); // Explicitly use CPU version

    // Draw circles only in the ROI
    for (const auto& contour : contours) {
        Point2f center;
        float radius;
        minEnclosingCircle(contour, center, radius);

        // Only draw circles if the center of the change is inside the ROI
        if (roi.contains(center)) {
            circle(img1_color, center, static_cast<int>(radius), Scalar(0, 0, 255), 2); // Red circle
        }
    }

    // Step 6: Save and Display Results
    imwrite("highlighted_changes_cuda.png", img1_color);
    imshow("Highlighted Changes", img1_color);
    waitKey(0);
    destroyAllWindows();

    return 0;
}

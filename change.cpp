#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    // Load binary images
    Mat img1 = imread("1.png", IMREAD_GRAYSCALE);
    Mat img2 = imread("2.jpg", IMREAD_GRAYSCALE);

    if (img1.empty() || img2.empty()) {
        cout << "Error: Could not load images." << endl;
        return -1;
    }

    // Ensure images are binary (convert to 0 and 1)
    threshold(img1, img1, 128, 1, THRESH_BINARY);
    threshold(img2, img2, 128, 1, THRESH_BINARY);

    // Step 1: Compute Difference Map (XOR operation)
    Mat diff;
    bitwise_xor(img1, img2, diff);

    // Step 2: Morphological Filtering to Remove Noise
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3)); // 3x3 kernel
    Mat filtered;
    morphologyEx(diff, filtered, MORPH_OPEN, kernel);

    // Step 3: Exclude 10% from Each Side
    int margin_x = static_cast<int>(img1.cols * 0.3); // 10% of width
    int margin_y = static_cast<int>(img1.rows * 0.1); // 10% of height

    // Define the central ROI, removing 10% from all four sides
    Rect roi(margin_x, margin_y, img1.cols - 2 * margin_x, img1.rows - 2 * margin_y);

    // Create a mask to exclude the border region
    Mat mask = Mat::zeros(filtered.size(), CV_8UC1);
    mask(roi).setTo(255); // Set only the inner region to white (255)

    // Apply the mask to keep only the central area
    Mat filtered_roi;
    bitwise_and(filtered, mask, filtered_roi);

    // Step 4: Connected Component Analysis
    Mat labels, stats, centroids;
    int num_labels = connectedComponentsWithStats(filtered_roi, labels, stats, centroids);

    int min_size = 10;  // Minimum size threshold
    int max_size = 20; // Maximum size threshold

    for (int i = 1; i < num_labels; i++) {  // Skip background (label 0)
        int area = stats.at<int>(i, CC_STAT_AREA);
        if (area < min_size || area > max_size) {
            filtered_roi.setTo(0, labels == i);
        }
    }

    // Step 5: Find Contours to Highlight Changes
    vector<vector<Point>> contours;
    findContours(filtered_roi, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // Convert img1 to color for visualization
    Mat img1_color;
    cvtColor(img1 * 255, img1_color, COLOR_GRAY2BGR);

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
    imwrite("highlighted_changes.png", img1_color);
    imshow("Highlighted Changes", img1_color);
    waitKey(0);
    destroyAllWindows();

    return 0;
}

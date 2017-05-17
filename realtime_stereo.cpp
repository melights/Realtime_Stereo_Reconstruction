/*
 *  stereo_match.cpp
 *  calibration
 *
 *  Created by Victor  Eruhimov on 1/18/10.
 *  Copyright 2010 Argus Corp. All rights reserved.
 *
 */

#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/cudastereo.hpp"
#include <pcl/common/common_headers.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <boost/thread/thread.hpp>
#include <stdio.h>
#include "StereoEfficientLargeScale.h"
using namespace cv;

Mat compute_disparity_ELAS(Mat &left, Mat &right)
{
    Mat disp;
    StereoEfficientLargeScale elas(0, 228);
    elas(left, right, disp, 100);
    return disp;
}

Mat compute_disparity_BM(Mat &left, Mat &right)
{
    Mat disp;
    Ptr<StereoBM> bm = StereoBM::create(48, 15);
    bm->compute(left, right, disp);
    return disp;
}

Mat compute_disparity_BM_GPU(Mat &left, Mat &right)
{
    cuda::GpuMat G_left, G_right, G_disp;
    Mat disp;
    Ptr<cuda::StereoBM> bm = cuda::createStereoBM(96, 25);
    G_left.upload(left);
    G_right.upload(right);
    bm->compute(G_left, G_right, G_disp);
    G_disp.download(disp);
    return disp;
}

int main(int argc, char **argv)
{
    Mat imageL, imageR, disp, disp8;
    char filenameL[500],filenameR[500];
    int imageNum = 1;
    int adder = 1;
    double minVal;
    double maxVal;
    namedWindow("left");
    namedWindow("disparity");
    while (1)
    {
        if (imageNum == 1237)
            adder = -1;
        else if (imageNum == 1)
            adder = 1;
        sprintf(filenameL, "/home/long/data/scale/right_rect/%04d.png", imageNum);
        sprintf(filenameR, "/home/long/data/scale/left_rect/%04d.png", imageNum);
        //std::cout << filenameL << std::endl;
        imageL = cv::imread(filenameL, 0);
        imageR = cv::imread(filenameR, 0);
        imageNum += adder;

        int64 t = getTickCount();

        disp = compute_disparity_BM(imageL, imageR);
        //disp = compute_disparity_ELAS(img1,img2);

        t = getTickCount() - t;
        double time = t / getTickFrequency();
        printf("Image %d: %fms, FPS: %f \n", imageNum, time * 1000, 1 / time);

        minMaxLoc(disp, &minVal, &maxVal);
        disp.convertTo(disp8, CV_8UC1, 255 / (maxVal - minVal));

        imshow("left", imageL);

        imshow("disparity", disp8);
        waitKey(1);
    }

    return 0;
}

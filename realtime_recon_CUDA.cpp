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
boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
int rows,cols;
cv::Mat Q;
enum RefImage {LeftRefImage, RightRefImage};
struct CostVolumeParams {

    uint8_t min_disp;
    uint8_t max_disp;
    uint8_t num_disp_layers;
    uint8_t method; // 0 for AD, 1 for ZNCC
    uint8_t win_r;
    RefImage ref_img;

};

struct PrimalDualParams {

    uint32_t num_itr;

    float alpha;
    float beta;
    float epsilon;
    float lambda;
    float aux_theta;
    float aux_theta_gamma;

    /* With preconditoining, we don't need these. */
    float sigma;
    float tau;
    float theta;

};

cv::Mat stereoCalcu(int _m, int _n, float* _left_img, float* _right_img, CostVolumeParams _cv_params, PrimalDualParams _pd_params);

void reconstruction(Mat &disp, Mat &img1c)
{
    cv::Mat image3D;
    cv::reprojectImageTo3D(disp, image3D, Q);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
    for(int x = 0; x < image3D.cols; x++)
    {
        for(int y = 0; y < image3D.rows; y++)
        {
            cv::Vec3f point3D = image3D.at<cv::Vec3f>(y,x);
            cv::Vec3b pointRGB = img1c.at<cv::Vec3b>(y,x);
            pcl::PointXYZRGB basic_point;
            basic_point.x = point3D.val[0];
            basic_point.y = point3D.val[1];
            basic_point.z = point3D.val[2];
            basic_point.b = pointRGB.val[0];
            basic_point.g = pointRGB.val[1];
            basic_point.r = pointRGB.val[2];
            if(cvIsInf(point3D.val[0]) || cvIsInf(point3D.val[1]) || cvIsInf(point3D.val[2]))
                ;//
            else
            {
 //               cout << point3D.val[0] << " " << point3D.val[1] << " " << point3D.val[2] << endl ;
                point_cloud_ptr->points.push_back(basic_point);

            }
        }
    }

    point_cloud_ptr->width = (int)point_cloud_ptr->points.size();
    point_cloud_ptr->height = 1;

    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(point_cloud_ptr);
    viewer->removeAllPointClouds();
    viewer->addPointCloud<pcl::PointXYZRGB>(point_cloud_ptr, rgb, "reconstruction");
    viewer->spinOnce(10000);
}

Mat compute_disparity_ELAS(Mat &left, Mat &right)
{
    Mat disp;
    StereoEfficientLargeScale elas(0, 228);
    elas(left, right, disp, 100);
    return disp;
}

Mat compute_disparity_CUDA(Mat &left, Mat &right)
{
    Mat letf_32F, right_32F;
    left.convertTo(letf_32F, CV_32F);
    right.convertTo(right_32F, CV_32F);

    CostVolumeParams cv_params;
    cv_params.min_disp = 0;
    cv_params.max_disp = 64;
    cv_params.method = 1;
    cv_params.win_r = 11;
    cv_params.ref_img = LeftRefImage;

    PrimalDualParams pd_params;
    pd_params.num_itr = 150; // 500
    pd_params.alpha = 0.1; // 10.0 0.01
    pd_params.beta = 1.0; // 1.0
    pd_params.epsilon = 0.1; // 0.1
    pd_params.lambda = 1e-2; // 1e-3
    pd_params.aux_theta = 10; // 10
    pd_params.aux_theta_gamma = 1e-6; // 1e-6


    cv::Mat result = stereoCalcu(letf_32F.rows, letf_32F.cols, (float*)letf_32F.data, (float*)right_32F.data, cv_params, pd_params);
    // convert for [0,1] to [min_d, max_d]
    result.convertTo(result, CV_32F, cv_params.max_disp);
    return result;
}

Mat compute_disparity_BM(Mat &left, Mat &right)
{
    Mat disp;
    Ptr<StereoBM> bm = StereoBM::create(96, 55);
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
    FileStorage fs("../Q.xml", FileStorage::READ);
    if (!fs.isOpened())
    {
        printf("Failed to open file ../Q.xml");
        return -1;
    }
    fs["Q"] >> Q;
    viewer->setBackgroundColor(0, 0, 0);
    viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();

    Mat imageL, imageR,image1c , disp, disp8;
    char filenameL[500], filenameR[500];
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
        sprintf(filenameL, "/home/long/data/Dataset6/right_rect/%04d.png", imageNum);
        sprintf(filenameR, "/home/long/data/Dataset6/left_rect/%04d.png", imageNum);
        //std::cout << filenameL << std::endl;
int64 t_start = getTickCount();

        imageL = cv::imread(filenameL, 0);
        imageR = cv::imread(filenameR, 0);
        image1c = cv::imread(filenameR, 1);
        imageNum += adder;
        rows=imageL.rows;
        cols=imageL.cols;
int64 t_read = getTickCount();

        //disp = compute_disparity_BM(imageL, imageR);
        //disp = compute_disparity_ELAS(img1,img2);
        disp = compute_disparity_CUDA(imageL, imageR);
int64 t_dispar = getTickCount();
        minMaxLoc(disp, &minVal, &maxVal);
        disp.convertTo(disp8, CV_8UC1, 255 / (maxVal - minVal));
std::cout<<disp<<std::endl;
        imshow("left", imageL);

        imshow("disparity", disp8);
int64 t_imshow = getTickCount();
        waitKey(10);
        reconstruction(disp,image1c);
int64 t_recon = getTickCount();
int64 t = getTickCount() - t_start;
        double time = t / getTickFrequency();
        printf("Image %d: read %fms,disparity %fms, imshow %fms,recom %fms,total %fms, FPS: %f \n", imageNum, float(t_read-t_start)/getTickFrequency()*1000,float(t_dispar-t_read)/getTickFrequency()*1000,float(t_imshow-t_dispar)/getTickFrequency()*1000,float(t_recon-t_imshow)/getTickFrequency()*1000, time * 1000, 1 / time);
        //printf("Image %d: %fms, FPS: %f \n", imageNum, time * 1000, 1 / time);
    }
}

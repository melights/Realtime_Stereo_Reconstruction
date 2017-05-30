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
void reconstruction(Mat &disp, Mat &img1c)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
    double px, py, pz;
    uchar pr, pg, pb;
    for (int i = 0; i < rows; i++)
    {
        const short *disp_ptr = disp.ptr<short>(i);
        for (int j = 0; j < cols; j++)
        {
            double d = static_cast<double>(disp_ptr[j]) / 16;
            //double d = (double)disp.at<Vec3b>(i, j)[0];

            if (d == -1)
                continue; //Discard bad pixels

            pz = 37.5585 / d;
            px = (static_cast<double>(j) - 338.665467) * pz / 751.706129;
            py = (static_cast<double>(i) - 257.986032) * pz / 766.315580;
            //std::cout << d << " ";
            pcl::PointXYZRGB point;
            point.x = px;
            point.y = py;
            point.z = pz;
            point.b = img1c.at<Vec3b>(i, j)[0];
            point.g = img1c.at<Vec3b>(i, j)[1];
            point.r = img1c.at<Vec3b>(i, j)[2];
            point_cloud_ptr->points.push_back(point);
        }
    }

    point_cloud_ptr->width = (int)point_cloud_ptr->points.size();
    point_cloud_ptr->height = 1;

    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(point_cloud_ptr);
    viewer->removeAllPointClouds();
    viewer->addPointCloud<pcl::PointXYZRGB>(point_cloud_ptr, rgb, "reconstruction");
    viewer->spinOnce();
}

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
VideoCapture cap_l("/home/long/data/Dataset9/left.avi");
VideoCapture cap_r("/home/long/data/Dataset9/right.avi");
    viewer->setBackgroundColor(0, 0, 0);
    viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();

    Mat imageL, imageR,imageL_mono, imageR_mono,image1c , disp, disp8;
    char filenameL[500], filenameR[500];
    int imageNum = 1;
    int adder = 1;
    double minVal;
    double maxVal;
    namedWindow("left");
    namedWindow("disparity");
    while (1)
    {

        //std::cout << filenameL << std::endl;
int64 t_start = getTickCount();
        cap_l.grab();
        cap_r.grab();
        cap_l.retrieve(imageL);
        cap_r.retrieve(imageR);
        cvtColor( imageL, imageL_mono, CV_BGR2GRAY );
        cvtColor( imageR, imageR_mono, CV_BGR2GRAY );

        rows=imageL.rows;
        cols=imageL.cols;
int64 t_read = getTickCount();

        disp = compute_disparity_BM(imageL_mono, imageR_mono);
        //disp = compute_disparity_ELAS(img1,img2);

int64 t_dispar = getTickCount();

        minMaxLoc(disp, &minVal, &maxVal);
        disp.convertTo(disp8, CV_8UC1, 255 / (maxVal - minVal));

        imshow("left", imageL);

        imshow("disparity", disp8);
int64 t_imshow = getTickCount();
        waitKey(1);
        reconstruction(disp,imageL);
int64 t_recon = getTickCount();
int64 t = getTickCount() - t_start;
        double time = t / getTickFrequency();
        printf("Image %d: read %fms,disparity %fms, imshow %fms,recom %fms,total %fms, FPS: %f \n", imageNum, float(t_read-t_start)/getTickFrequency()*1000,float(t_dispar-t_read)/getTickFrequency()*1000,float(t_imshow-t_dispar)/getTickFrequency()*1000,float(t_recon-t_imshow)/getTickFrequency()*1000, time * 1000, 1 / time);
        //printf("Image %d: %fms, FPS: %f \n", imageNum, time * 1000, 1 / time);
    }
}

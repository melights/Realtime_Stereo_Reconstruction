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
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d.h>
#include <pcl/surface/gp3.h>

#include <boost/thread/thread.hpp>
#include <stdio.h>
#include "StereoEfficientLargeScale.h"
using namespace cv;
boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);
int rows, cols;

#define STRIDE 8

void GreedyProjectionTriangulation()
{
    // Normal estimation*
std::cout<<point_cloud_ptr->points.size()<<std::endl;
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> n;
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(point_cloud_ptr);
    n.setInputCloud(point_cloud_ptr);
    n.setSearchMethod(tree);
    n.setKSearch(20);
    n.compute(*normals);
    //* normals should not contain the point normals + surface curvatures
    // Concatenate the XYZ and normal fields*
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals(new pcl::PointCloud<pcl::PointNormal>);
    pcl::concatenateFields(*point_cloud_ptr, *normals, *cloud_with_normals);
    //* cloud_with_normals = cloud + normals
    // Create search tree*
    pcl::search::KdTree<pcl::PointNormal>::Ptr tree2(new pcl::search::KdTree<pcl::PointNormal>);
    tree2->setInputCloud(cloud_with_normals);
    // Initialize objects
    pcl::GreedyProjectionTriangulation<pcl::PointNormal> gp3;
    pcl::PolygonMesh triangles;

    // Set the maximum distance between connected points (maximum edge length)
    gp3.setSearchRadius(0.025);

    // Set typical values for the parameters
    gp3.setMu(2.5);
    gp3.setMaximumNearestNeighbors(100);
    gp3.setMaximumSurfaceAngle(M_PI / 4); // 45 degrees
    gp3.setMinimumAngle(M_PI / 18);       // 10 degrees
    gp3.setMaximumAngle(2 * M_PI / 3);    // 120 degrees
    gp3.setNormalConsistency(false);

    // Get result
    gp3.setInputCloud(cloud_with_normals);
    gp3.setSearchMethod(tree2);
    gp3.reconstruct(triangles);
    viewer->removePolygonMesh("my");
    viewer->addPolygonMesh(triangles, "my"); //设置所要显示的网格对象
    viewer->spinOnce();

}

void reconstruction_pointcloud(Mat &disp, Mat &img1c)
{
    double px, py, pz;
    uchar pr, pg, pb;
    pcl::PointCloud<pcl::PointXYZ>::Ptr tmp(new pcl::PointCloud<pcl::PointXYZ>);
    for (int i = 0; i < rows - STRIDE; i += STRIDE)
    {
        const short *disp_ptr = disp.ptr<short>(i);
        for (int j = 0; j < cols - STRIDE; j += STRIDE)
        {
            double d = static_cast<double>(disp_ptr[j]) / 16;
            //double d = (double)disp.at<Vec3b>(i, j)[0];

            if (d == -1)
                continue; //Discard bad pixels

            pz = 37.5585 / d;
            px = (static_cast<double>(j) - 338.665467) * pz / 751.706129;
            py = (static_cast<double>(i) - 257.986032) * pz / 766.315580;
            //std::cout << d << " ";
            pcl::PointXYZ point;
            point.x = px;
            point.y = py;
            point.z = pz;
            tmp->points.push_back(point);
        }
    }

     tmp->width = (int)tmp->points.size();
     tmp->height = 1;
     tmp->is_dense = false;
     point_cloud_ptr->swap( *tmp );

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

    viewer->setBackgroundColor(0, 0, 0);
    viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();

    Mat imageL, imageR, image1c, disp, disp8;
    char filenameL[500], filenameR[500];
    int imageNum = 1;
    int adder = 1;
    double minVal;
    double maxVal;
    namedWindow("left");
    namedWindow("disparity");
    while (1)
    {
        //////////Read Images/////////
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
        image1c = cv::imread(filenameL, 1);
        imageNum += adder;
        rows = imageL.rows;
        cols = imageL.cols;
        int64 t_read = getTickCount();

        //////////Compute Disparity Map/////////
        disp = compute_disparity_BM(imageL, imageR);
        //disp = compute_disparity_BM_GPU(imageL, imageR);
        //disp = compute_disparity_ELAS(imageL,imageR);
        int64 t_dispar = getTickCount();

        //////////Display Image & Disparity Map/////////
        minMaxLoc(disp, &minVal, &maxVal);
        disp.convertTo(disp8, CV_8UC1, 255 / (maxVal - minVal));
        imshow("left", imageL);
        imshow("disparity", disp8);
        int64 t_imshow = getTickCount();
        waitKey(1);

        //////////Point Cloud Reconstruction/////////
        //point_cloud_ptr.reset();
        reconstruction_pointcloud(disp, image1c);
        int64 t_recon = getTickCount();

        //////////Surface Reconstruction/////////
        GreedyProjectionTriangulation();
        int64 t_surface = getTickCount();


        //////////Time/////////
        int64 t = getTickCount() - t_start;
        double time = t / getTickFrequency();
        printf("Image %d: read %fms, disparity %fms, imshow %fms, recon %fms, surface %fms, total %fms, FPS: %f \n", 
        imageNum, 
        float(t_read - t_start) / getTickFrequency() * 1000,     //Read
        float(t_dispar - t_read) / getTickFrequency() * 1000,   //Disparity
        float(t_imshow - t_dispar) / getTickFrequency() * 1000, //Imshow
        float(t_recon - t_imshow) / getTickFrequency() * 1000,  //Recon
        float(t_surface - t_recon) / getTickFrequency() * 1000, //Surface
        time * 1000,  //Total
        1 / time); //FPS

    }
}

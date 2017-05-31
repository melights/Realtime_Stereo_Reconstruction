/* For the algorithm details, please refer to:
     [1] Ping-Lin Chang, Danail Stoyanov, Andrew J. Davison, Philip "Eddie" Edwards.
     Real-Time Dense Stereo Reconstruction Using Convex Optimisation with a Cost-Volume
     for Image-Guided Robotic Surgery, MICCAI, 2013.
     Email: p.chang10@imperial.ac.uk
*/
// CUDA-C includes
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/cudev/common.hpp>
#include <opencv2/cudaarithm.hpp>

#include "cuda_interface.cuh"
#include "time.h"

using namespace cv;

extern "C"
//Adds two arrays
void runCudaPart(cuda::PtrStepSz<float> data, cuda::PtrStepSz<float> result,int rows, int cols);
void runCudaSet(float* data, float* result, int rows, int cols);

cv::Mat stereoCalcu(int _m, int _n, float* _left_img, float* _right_img, CostVolumeParams _cv_params, PrimalDualParams _pd_params);


__global__ void addAry( int * ary1, int * ary2 )
{
    int indx = threadIdx.x;
    //ary1[ indx ] += ary2[ indx ];
    ary1[ indx ] = 1;
}

__global__ void changetoone(cuda::PtrStepSz<float> data, cuda::PtrStepSz<float> result)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;
    //int offset = x + y*gridDim.x;
    //int offset = threadIdx.x + blockIdx.x * blockDim.x;


    if(offset < 640*480)
        if(offset > 640 && (offset+640) < 640*480)
            result(0, offset) = data(0, offset) - data(0, offset-1);

}

__global__ void ChangeToOne(float* data, float* result)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;
    //int offset = x + y*gridDim.x;
    //int offset = threadIdx.x + blockIdx.x * blockDim.x;

    if(offset < 640*480)
        if(offset > 640 && (offset+640) < 640*480)
            result[offset] = data[offset] - data[offset-1];
}


// Main cuda function
void runCudaSet(float* data, float* result, int rows, int cols)
{

    printf( "start\n %d %d\n", rows, cols);
    dim3 blocks((cols+31)/32, (rows+31)/32);
    dim3 threads(32,32);
    ChangeToOne<<<blocks, threads>>>(data, result);
}

void runCudaPart(cuda::PtrStepSz<float> data, cuda::PtrStepSz<float> result,int rows, int cols)
{
    printf( "start\n %d %d\n", rows, cols);
    dim3 blocks((cols+31)/32, (rows+31)/32);
    dim3 threads(32,32);
    changetoone<<<blocks, threads>>>(data, result);

}

cv::Mat stereoCalcu(int _rows, int _cols, float* _left_img, float* _right_img, CostVolumeParams _cv_params, PrimalDualParams _pd_params)
{
    clock_t start, finish;
    double duration = 0;
//    start = clock();
//    finish = clock();
//    double duration = (double)(finish - start) / CLOCKS_PER_SEC;
//    printf("%f \n", duration);


    ////////////////////////////////////////////////////////////////////////////
    //  Check and init the input/output variables.
    ////////////////////////////////////////////////////////////////////////////

    size_t width = _cols;
    size_t height = _rows;

    CostVolumeParams host_cv_params;
    host_cv_params.min_disp = _cv_params.min_disp;
    host_cv_params.max_disp = _cv_params.max_disp;
    host_cv_params.num_disp_layers = host_cv_params.max_disp-host_cv_params.min_disp+1;

    host_cv_params.method = _cv_params.method;
    host_cv_params.win_r = _cv_params.win_r;
    host_cv_params.ref_img = _cv_params.ref_img;

    CostVolumeParams* dev_cv_params;
    checkCudaErrors(cudaMalloc((void**)&dev_cv_params, sizeof(CostVolumeParams)));
    checkCudaErrors(cudaMemcpy(dev_cv_params, &host_cv_params, sizeof(CostVolumeParams), cudaMemcpyHostToDevice));

    printf("Cost Volume params set.\n");

    /* Check primal dual parameters */

    PrimalDualParams host_pd_params;
    host_pd_params.num_itr = _pd_params.num_itr;
    host_pd_params.alpha = _pd_params.alpha;
    host_pd_params.beta = _pd_params.beta;
    host_pd_params.epsilon = _pd_params.epsilon;
    host_pd_params.lambda = _pd_params.lambda;
    host_pd_params.aux_theta = _pd_params.aux_theta;
    host_pd_params.aux_theta_gamma = _pd_params.aux_theta_gamma;
    host_pd_params.theta = 1.0;

    PrimalDualParams* dev_pd_params;
    checkCudaErrors(cudaMalloc((void**)&dev_pd_params, sizeof(PrimalDualParams)));
    checkCudaErrors(cudaMemcpy(dev_pd_params, &host_pd_params, sizeof(PrimalDualParams), cudaMemcpyHostToDevice));

    printf("Primal dual params set.\n");

    /* Allocate device memory and copy left and right image. */
    cudaArray *left_img_array, *right_img_array;
    cudaChannelFormatDesc channelDesc_float = cudaCreateChannelDesc<float>();
    checkCudaErrors(cudaMallocArray(&left_img_array, &channelDesc_float, width, height));
    checkCudaErrors(cudaMallocArray(&right_img_array, &channelDesc_float, width, height));
    checkCudaErrors(cudaMemcpyToArray(left_img_array, 0, 0, (float*)_left_img, width*height*sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyToArray(right_img_array, 0, 0, (float*)_right_img, width*height*sizeof(float), cudaMemcpyHostToDevice));

    printf("Allocate device memory and copy left and right image.\n");

    /* Allocate cost-volume 3D memory. */
    cudaPitchedPtr cost_volume;
    checkCudaErrors(cudaMalloc3D(&cost_volume, make_cudaExtent(width*sizeof(float), height, host_cv_params.num_disp_layers)));

    printf("Allocate cost-volume 3D memory.\n");

    ////////////////////////////////////////////////////////////////////////////
    //  Cost-volume building
    ////////////////////////////////////////////////////////////////////////////

    /* Bind to read-only textures */
    if(host_cv_params.ref_img == LeftRefImage)
    {
        checkCudaErrors(cudaBindTextureToArray(ref_img_tex, left_img_array));
        checkCudaErrors(cudaBindTextureToArray(target_img_tex, right_img_array));
    }
    else if(host_cv_params.ref_img == RightRefImage)
    {
        checkCudaErrors(cudaBindTextureToArray(ref_img_tex, right_img_array));
        checkCudaErrors(cudaBindTextureToArray(target_img_tex, left_img_array));
    }

    size_t THREAD_NUM_3D_BLOCK = 8;
    dim3 dimBlock(THREAD_NUM_3D_BLOCK, THREAD_NUM_3D_BLOCK,THREAD_NUM_3D_BLOCK);
    dim3 dimGrid((width+dimBlock.x-1)/dimBlock.x,
                 (height+dimBlock.y-1)/dimBlock.y,
                 (host_cv_params.num_disp_layers+dimBlock.z-1)/dimBlock.z);

    start = clock();

    /* AD */
    if(host_cv_params.method == 0)
    {
        ADKernel<<<dimGrid, dimBlock>>>(cost_volume,
                                        dev_cv_params,
                                        width,
                                        height);
        printf("AD done!\n");
    }
    /* ZNCC */
    if(host_cv_params.method == 1)
    {
        ZNCCKernel<<<dimGrid, dimBlock>>>(cost_volume,
                                          dev_cv_params,
                                          width,
                                          height);
        printf("ZNCC done!\n");
    }

    finish = clock();
    duration = (double)(finish - start) / CLOCKS_PER_SEC;
    printf("It take %f s.\n", duration);

    /* Copy cost-volume to 3D array and bind it to 3D texture for fast accessing */
    cudaArray* cost_volume_array;
    checkCudaErrors(cudaMalloc3DArray(&cost_volume_array, &channelDesc_float, make_cudaExtent(width, height, host_cv_params.num_disp_layers), cudaArraySurfaceLoadStore));

    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr = cost_volume;
    copyParams.dstArray = cost_volume_array;
    copyParams.extent = make_cudaExtent(width, height, host_cv_params.num_disp_layers);
    copyParams.kind   = cudaMemcpyDeviceToDevice;
    checkCudaErrors(cudaMemcpy3D(&copyParams));

    checkCudaErrors(cudaBindTextureToArray(cost_volume_tex, cost_volume_array, channelDesc_float));
    checkCudaErrors(cudaFree(cost_volume.ptr));

    ////////////////////////////////////////////////////////////////////////////
    //  Winnder-take-all (WTA) scheme to initialise disp
    ////////////////////////////////////////////////////////////////////////////
    cudaPitchedPtr min_disp, min_disp_cost, max_disp_cost;
    checkCudaErrors(cudaMallocPitch((void **)&min_disp.ptr, &min_disp.pitch, width*sizeof(Primal), height));
    checkCudaErrors(cudaMemset2D(min_disp.ptr, min_disp.pitch, 0.0, width*sizeof(Primal), height));

    checkCudaErrors(cudaMallocPitch((void **)&min_disp_cost.ptr, &min_disp_cost.pitch, width*sizeof(float), height));
    checkCudaErrors(cudaMemset2D(min_disp_cost.ptr, min_disp_cost.pitch, 0.0, width*sizeof(float), height));

    checkCudaErrors(cudaMallocPitch((void **)&max_disp_cost.ptr, &max_disp_cost.pitch, width*sizeof(float), height));
    checkCudaErrors(cudaMemset2D(max_disp_cost.ptr, max_disp_cost.pitch, 0.0, width*sizeof(float), height));

    size_t THREAD_NUM_2D_BLOCK = 16;
    dimBlock = dim3(THREAD_NUM_2D_BLOCK, THREAD_NUM_2D_BLOCK);
    dimGrid = dim3((width+dimBlock.x-1)/dimBlock.x,
                   (height+dimBlock.y-1)/dimBlock.y);


    start = clock();
    WTAKernel<<<dimGrid, dimBlock>>>(min_disp,
                                     min_disp_cost,
                                     max_disp_cost,
                                     dev_cv_params,
                                     width,
                                     height);

    finish = clock();
    duration = (double)(finish - start) / CLOCKS_PER_SEC;
    printf("WTAKernel take %f s.\n", duration);

    checkCudaErrors(cudaBindTexture2D(0, min_disp_cost_tex, min_disp_cost.ptr,
                                      cudaCreateChannelDesc<float>(),
                                      width, height, min_disp_cost.pitch));

    checkCudaErrors(cudaBindTexture2D(0, max_disp_cost_tex, max_disp_cost.ptr,
                                      cudaCreateChannelDesc<float>(),
                                      width, height, max_disp_cost.pitch));


     // return first
     // Mat result(_rows, _cols, CV_32F);
     // checkCudaErrors(cudaMemcpy2D((float*)result.data, width*sizeof(float), min_disp.ptr, min_disp.pitch, width*sizeof(float), height, cudaMemcpyDeviceToHost));
     // return result;

    ////////////////////////////////////////////////////////////////////////////
    //  Primal-dual + cost-volume optimisation
    ////////////////////////////////////////////////////////////////////////////

    /* Allocate and initialise variables */
    cudaPitchedPtr primal, old_primal, head_primal, dual, aux, diffuse_tensor, error_img;
    cudaPitchedPtr primal_step, dual_step;

    /* Primal variables */
    checkCudaErrors(cudaMallocPitch((void **)&primal.ptr, &primal.pitch, width*sizeof(Primal), height));
    checkCudaErrors(cudaMemcpy2D(primal.ptr, primal.pitch, min_disp.ptr, min_disp.pitch, width*sizeof(Primal), height, cudaMemcpyDeviceToDevice));

    checkCudaErrors(cudaMallocPitch((void **)&head_primal.ptr, &head_primal.pitch, width*sizeof(Primal), height));
    checkCudaErrors(cudaMemcpy2D(head_primal.ptr, head_primal.pitch, min_disp.ptr, min_disp.pitch, width*sizeof(Primal), height, cudaMemcpyDeviceToDevice));

    checkCudaErrors(cudaMallocPitch((void **)&old_primal.ptr, &old_primal.pitch, width*sizeof(Primal), height));

    checkCudaErrors(cudaMallocPitch((void **)&primal_step.ptr, &primal_step.pitch, width*sizeof(PrimalStep), height));
    checkCudaErrors(cudaMemset2D(primal_step.ptr, primal_step.pitch, 0.0, width*sizeof(PrimalStep), height));

    /* Dual variables */
    checkCudaErrors(cudaMallocPitch((void **)&dual.ptr, &dual.pitch, width*sizeof(Dual), height));
    checkCudaErrors(cudaMemset2D(dual.ptr, dual.pitch, 0.0, width*sizeof(Dual), height));

    checkCudaErrors(cudaMallocPitch((void **)&dual_step.ptr, &dual_step.pitch, width*sizeof(DualStep), height));
    checkCudaErrors(cudaMemset2D(dual_step.ptr, dual_step.pitch, 0.0, width*sizeof(DualStep), height));

    /* Auxiliary variable */
    checkCudaErrors(cudaMallocPitch((void **)&aux.ptr, &aux.pitch, width*sizeof(Auxiliary), height));
    checkCudaErrors(cudaMemcpy2D(aux.ptr, aux.pitch, min_disp.ptr, min_disp.pitch, width*sizeof(Auxiliary), height, cudaMemcpyDeviceToDevice));

    /* Weighting matrix using 2x2 D tensor matrix */
    checkCudaErrors(cudaMallocPitch((void **)&diffuse_tensor.ptr, &diffuse_tensor.pitch, width*sizeof(DiffuseTensor), height));
    checkCudaErrors(cudaMemset2D(diffuse_tensor.ptr, diffuse_tensor.pitch, 0.0, width*sizeof(DiffuseTensor), height));

    /* Point-wise errors */
    checkCudaErrors(cudaMallocPitch((void **)&error_img.ptr, &error_img.pitch, width*sizeof(float), height));
    checkCudaErrors(cudaMemset2D(error_img.ptr, error_img.pitch, 0.0, width*sizeof(float), height));

    /* Calculating diffusion tensor */
    DiffuseTensorKernel<<<dimGrid, dimBlock>>>(diffuse_tensor,
                                               dev_pd_params,
                                               dev_cv_params,
                                               width,
                                               height);

    /* Bind textures */
    checkCudaErrors(cudaBindTexture2D(0, diffuse_tensor_tex, diffuse_tensor.ptr,
                                      cudaCreateChannelDesc<float4>(),
                                      width, height, diffuse_tensor.pitch));

    checkCudaErrors(cudaBindTexture2D(0, head_primal_tex, head_primal.ptr,
                                      cudaCreateChannelDesc<float>(),
                                      width, height, head_primal.pitch));

    checkCudaErrors(cudaBindTexture2D(0, primal_tex, primal.ptr,
                                      cudaCreateChannelDesc<float>(),
                                      width, height, primal.pitch));

    checkCudaErrors(cudaBindTexture2D(0, old_primal_tex, old_primal.ptr,
                                      cudaCreateChannelDesc<float>(),
                                      width, height, old_primal.pitch));

    checkCudaErrors(cudaBindTexture2D(0, dual_tex, dual.ptr,
                                      cudaCreateChannelDesc<float2>(),
                                      width, height, dual.pitch));

    checkCudaErrors(cudaBindTexture2D(0,
                                      aux_tex,
                                      aux.ptr,
                                      cudaCreateChannelDesc<float>(),
                                      width,
                                      height,
                                      aux.pitch));

    checkCudaErrors(cudaBindTexture2D(0,
                                      primal_step_tex,
                                      primal_step.ptr,
                                      cudaCreateChannelDesc<float>(),
                                      width,
                                      height,
                                      primal_step.pitch));

    checkCudaErrors(cudaBindTexture2D(0,
                                      dual_step_tex,
                                      dual_step.ptr,
                                      cudaCreateChannelDesc<float2>(),
                                      width,
                                      height,
                                      dual_step.pitch));

    start = clock();

    /* Do preconditioning on the linear operator (D and nabla) */
    DiffusionPrecondKernel<<<dimGrid, dimBlock>>>(primal_step,
                                                  dual_step,
                                                  dev_cv_params,
                                                  width,
                                                  height);

    float errors[host_pd_params.num_itr];
    float host_error_img[width*height];
    for(uint32_t i = 0; i < host_pd_params.num_itr; i++)
    {

        checkCudaErrors(cudaMemcpy2D(old_primal.ptr,
                                     old_primal.pitch,
                                     primal.ptr,
                                     primal.pitch,
                                     width*sizeof(Primal),
                                     height,
                                     cudaMemcpyDeviceToDevice));

        /* Dual update */
        HuberL2DualPrecondKernel<<<dimGrid, dimBlock>>>(dual,
                                                        dev_pd_params,
                                                        dev_cv_params,
                                                        width,
                                                        height);

        /* Primal update */
        HuberL2PrimalPrecondKernel<<<dimGrid, dimBlock>>>(primal,
                                                          dev_pd_params,
                                                          dev_cv_params,
                                                          width,
                                                          height);

        /* Head primal update */
        HuberL2HeadPrimalKernel<<<dimGrid, dimBlock>>>(head_primal,
                                                       dev_pd_params,
                                                       dev_cv_params,
                                                       width,
                                                       height);

        /* Pixel-wise line search in cost-volume */
        CostVolumePixelWiseSearch<<<dimGrid, dimBlock>>>(aux,
                                                         dev_pd_params,
                                                         dev_cv_params,
                                                         width,
                                                         height);

        host_pd_params.aux_theta = host_pd_params.aux_theta*(1.0 - host_pd_params.aux_theta_gamma*i);
        checkCudaErrors(cudaMemcpy(&dev_pd_params->aux_theta, &host_pd_params.aux_theta, sizeof(float), cudaMemcpyHostToDevice));

        /* Calculate point-wise error */
        HuberL1CVErrorKernel<<<dimGrid, dimBlock>>>(error_img,
                                                    dev_pd_params,
                                                    dev_cv_params,
                                                    width,
                                                    height);

        checkCudaErrors(cudaMemcpy2D(host_error_img, width*sizeof(float),
                                     error_img.ptr, error_img.pitch,
                                     width*sizeof(float), height, cudaMemcpyDeviceToHost));

        errors[i] = 0.0;
        for(uint32_t e = 0; e < width*height; e++)
            errors[i] += host_error_img[e];

    }

    finish = clock();
    duration = (double)(finish - start) / CLOCKS_PER_SEC;
    printf("Optimiztion take %f s.\n", duration);

    //////////////////////////////////////////////////////////////////////////
    // Copy back
    //////////////////////////////////////////////////////////////////////////

    Mat result(_rows, _cols, CV_32F);
    checkCudaErrors(cudaMemcpy2D((float*)result.data, width*sizeof(Primal), primal.ptr, primal.pitch, width*sizeof(Primal), height, cudaMemcpyDeviceToHost));
    //result.reshape(0, _rows);

    /* Free CUDA memory */

    checkCudaErrors(cudaUnbindTexture(ref_img_tex));
    checkCudaErrors(cudaUnbindTexture(target_img_tex));

    checkCudaErrors(cudaUnbindTexture(cost_volume_tex));

    checkCudaErrors(cudaUnbindTexture(min_disp_cost_tex));
    checkCudaErrors(cudaUnbindTexture(max_disp_cost_tex));


    checkCudaErrors(cudaUnbindTexture(aux_tex));

    checkCudaErrors(cudaUnbindTexture(head_primal_tex));
    checkCudaErrors(cudaUnbindTexture(old_primal_tex));
    checkCudaErrors(cudaUnbindTexture(primal_tex));
    checkCudaErrors(cudaUnbindTexture(primal_step_tex));

    checkCudaErrors(cudaUnbindTexture(dual_tex));
    checkCudaErrors(cudaUnbindTexture(dual_step_tex));

    checkCudaErrors(cudaUnbindTexture(diffuse_tensor_tex));

    checkCudaErrors(cudaFree(min_disp.ptr));
    checkCudaErrors(cudaFree(min_disp_cost.ptr));
    checkCudaErrors(cudaFree(max_disp_cost.ptr));

    checkCudaErrors(cudaFree(primal.ptr));
    checkCudaErrors(cudaFree(old_primal.ptr));
    checkCudaErrors(cudaFree(head_primal.ptr));

    checkCudaErrors(cudaFree(dual.ptr));
    checkCudaErrors(cudaFree(aux.ptr));
    checkCudaErrors(cudaFree(diffuse_tensor.ptr));
    checkCudaErrors(cudaFree(error_img.ptr));

    checkCudaErrors(cudaFree(primal_step.ptr));
    checkCudaErrors(cudaFree(dual_step.ptr));

    checkCudaErrors(cudaFreeArray(cost_volume_array));
    checkCudaErrors(cudaFreeArray(left_img_array));
    checkCudaErrors(cudaFreeArray(right_img_array));

    return result;
}

/* For the algorithm details, please refer to:
     [1] Ping-Lin Chang, Danail Stoyanov, Andrew J. Davison, Philip "Eddie" Edwards.
     Real-Time Dense Stereo Reconstruction Using Convex Optimisation with a Cost-Volume
     for Image-Guided Robotic Surgery, MICCAI, 2013.
     Email: p.chang10@imperial.ac.uk
*/
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <stdint.h>
#include <string>

using namespace std;

/* CUDA texutres */
texture<float, cudaTextureType2D> ref_img_tex;
texture<float, cudaTextureType2D> target_img_tex;

texture<float, cudaTextureType3D> cost_volume_tex;

texture<float, cudaTextureType2D> min_disp_cost_tex;
texture<float, cudaTextureType2D> max_disp_cost_tex;

texture<float, cudaTextureType2D> aux_tex;

texture<float, cudaTextureType2D> head_primal_tex;
texture<float, cudaTextureType2D> old_primal_tex;
texture<float, cudaTextureType2D> primal_tex;
texture<float, cudaTextureType2D> primal_step_tex;

texture<float2, cudaTextureType2D> dual_tex;
texture<float2, cudaTextureType2D> dual_step_tex;

texture<float4, cudaTextureType2D> diffuse_tensor_tex;

#define cost_max 1.0

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

struct Primal { float u; };
struct PrimalStep { float tau; };

struct Dual { float p[2]; };
struct DualStep { float sigma[2]; };

struct Auxiliary { float a; };
struct DiffuseTensor { float D[4]; };

/*
 *  Stereo matching kernels
 *
 *  Remark: Bear in mind that Matlab memory is column-major and C/C++ CUDA memory is row-major (swap x and y).
 */

/* Absolute Differences (AD) */
void __global__ ADKernel(cudaPitchedPtr const cost_volume,
                         CostVolumeParams const *const cv_params,
                         size_t const width,
                         size_t const height)
{
    uint32_t const x = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t const y = blockDim.y * blockIdx.y + threadIdx.y;
    uint32_t const z = blockDim.z * blockIdx.z + threadIdx.z;

    uint8_t const min_disp = cv_params->min_disp;
    uint8_t const max_disp = cv_params->max_disp;
    uint8_t const num_disp_laysers = cv_params->num_disp_layers;
    RefImage const ref_img = cv_params->ref_img;

    if(x < width && y < height && z < num_disp_laysers)
    {
        char* volume_ptr = (char*)cost_volume.ptr;
        uint32_t volume_pitch = cost_volume.pitch;
        uint32_t volume_slice_pitch = volume_pitch * height;

        char* volume_slice = volume_ptr + z * volume_slice_pitch;
        float* volume_row = (float*)(volume_slice + y * volume_pitch);

        volume_row[x] = cost_max;

        uint8_t disp = z+min_disp;
        switch (ref_img)
        {
        case LeftRefImage:
            if(x >= max_disp)
                //volume_row[x] = abs(tex2D(ref_img_tex, x, y) - tex2D(target_img_tex, x+0.5, y-disp));
                volume_row[x] = abs(tex2D(ref_img_tex, x, y) - tex2D(target_img_tex, x-disp, y+0.5));
            break;
        case RightRefImage:
            if(x < height-max_disp)
                volume_row[x] = abs(tex2D(ref_img_tex, x, y) - tex2D(target_img_tex, x+disp, y+0.5));
                //volume_row[x] = abs(tex2D(ref_img_tex, x, y) - tex2D(target_img_tex, x+0.5, y+disp));
            break;
        default:
            break;
        }
    }
}

/* Zero norm_grad_imgalised Cross-Correlation (ZNCC) */
void __global__ ZNCCKernel(cudaPitchedPtr const cost_volume,
                           CostVolumeParams const *const cv_params,
                           size_t const width,
                           size_t const height)
{
    uint32_t const x = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t const y = blockDim.y * blockIdx.y + threadIdx.y;
    uint32_t const z = blockDim.z * blockIdx.z + threadIdx.z;

    uint8_t const min_disp = cv_params->min_disp;
    uint8_t const max_disp = cv_params->max_disp;
    uint8_t const num_disp_laysers = cv_params->num_disp_layers;
    uint8_t const win_r    = cv_params->win_r;
    RefImage const ref_img = cv_params->ref_img;

    if (x < width && y < height && z < num_disp_laysers)
    {
        char* volume_ptr = (char*)cost_volume.ptr;
        uint32_t volume_pitch = cost_volume.pitch;
        uint32_t volume_slice_pitch = volume_pitch * height;

        char* volume_slice = volume_ptr + z * volume_slice_pitch;
        float* volume_row = (float*)(volume_slice + y * volume_pitch);

        volume_row[x] = cost_max;

        uint8_t disp = z+min_disp;
        double sum_ref = 0, sum_target = 0, sum_ref_target = 0, sum_sq_ref = 0, sum_sq_target = 0;

        switch (ref_img)
        {
        case LeftRefImage:
            if(x >= max_disp+win_r && x < width-win_r && y >= win_r && y < height-win_r)
            {
                for(int win_x = x-win_r; win_x <= x+win_r; win_x++)
                    for(int win_y = y-win_r; win_y <= y+win_r; win_y++)
                    {
                        double ref_gray = tex2D(ref_img_tex, win_x, win_y);
                        double target_gray = tex2D(target_img_tex, win_x-disp, win_y);

                        sum_ref += ref_gray;
                        sum_target += target_gray;

                        sum_ref_target += ref_gray*target_gray;

                        sum_sq_ref += ref_gray*ref_gray;
                        sum_sq_target += target_gray*target_gray;
                    }

                double N = (2*win_r+1)*(2*win_r+1);

                double numerator = N*sum_ref_target - sum_ref*sum_target;
                double denominator = (N*sum_sq_target - sum_target*sum_target)*(N*sum_sq_ref - sum_ref*sum_ref);

                if(denominator > 0)
                    volume_row[x] = 1.0 - (numerator*rsqrtf(abs(denominator))+1.0)/2.0;


            }
            break;
        case RightRefImage:
            if(x >= win_r && x < width-max_disp-win_r && y >= win_r && y < height-win_r)
            {
                for(int win_x = x-win_r; win_x <= x+win_r; win_x++)
                    for(int win_y = y-win_r; win_y <= y+win_r; win_y++)
                    {
                        float ref_gray = tex2D(ref_img_tex, win_x, win_y);
                        float target_gray = tex2D(target_img_tex, win_x+disp, win_y);

                        sum_ref += ref_gray;
                        sum_target += target_gray;

                        sum_ref_target += ref_gray*target_gray;

                        sum_sq_ref += ref_gray*ref_gray;
                        sum_sq_target += target_gray*target_gray;
                    }

                double N = (2*win_r+1)*(2*win_r+1);

                double numerator = N*sum_ref_target - sum_ref*sum_target;
                double denominator = (N*sum_sq_target - sum_target*sum_target)*(N*sum_sq_ref - sum_ref*sum_ref);

                if(denominator > 0)
                    volume_row[x] = 1.0 - (numerator*rsqrtf(abs(denominator))+1.0)/2.0;

            }
            break;
        default:
            break;
        }

    }

}

/*
 *  Winner-take-all kernel
 *
 *  Remark: The cost value is norm_grad_imgalised to [0,1]
 */

void __global__ WTAKernel(cudaPitchedPtr const min_disp,
                          cudaPitchedPtr const min_disp_cost,
                          cudaPitchedPtr const max_disp_cost,
                          CostVolumeParams const *const cv_params,
                          size_t const width,
                          size_t const height)
{
    uint32_t const x = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t const y = blockDim.y * blockIdx.y + threadIdx.y;

    uint8_t const max_disp = cv_params->max_disp;
    uint8_t const win_r = cv_params->win_r;

    uint8_t const num_disp_layers = cv_params->num_disp_layers;

    switch (cv_params->ref_img)
    {
    case LeftRefImage:
        if(x >= max_disp+win_r && x < width-win_r && y >= win_r && y < height-win_r)
        {

            float min_cost = cost_max;
            uint32_t min_cost_idx = 0;
            for(uint32_t z = 0; z < num_disp_layers; z++)
            {

                if(tex3D(cost_volume_tex, x, y, z) < min_cost)
                {
                    min_cost = tex3D(cost_volume_tex, x, y, z);
                    min_cost_idx = z;
                }

            }

            Primal* disp_row = (Primal*)((char*)min_disp.ptr + y*min_disp.pitch);
            disp_row[x].u = float(min_cost_idx)/float(num_disp_layers-1);

            float* min_disp_cost_row = (float*)((char*)min_disp_cost.ptr + y*min_disp_cost.pitch);
            min_disp_cost_row[x] = min_cost;

            /* Search for max cost value */
            float max_cost = -1;
            for(uint32_t z = 0; z < num_disp_layers; z++)
                if(tex3D(cost_volume_tex, x, y, z) > max_cost)
                    max_cost = tex3D(cost_volume_tex, x, y, z);

            float* max_disp_cost_row = (float*)((char*)max_disp_cost.ptr + y*max_disp_cost.pitch);
            max_disp_cost_row[x] = max_cost;

        }
        break;
    case RightRefImage:
        if(x >= win_r && x < width-max_disp-win_r && y >= win_r && y < height-win_r)
        {

            float min_cost = cost_max;
            uint32_t min_cost_idx = 0;
            for(uint32_t z = 0; z < num_disp_layers; z++)
            {

                if(tex3D(cost_volume_tex, x, y, z) < min_cost)
                {
                    min_cost = tex3D(cost_volume_tex, x, y, z);
                    min_cost_idx = z;
                }

            }

            Primal* disp_row = (Primal*)((char*)min_disp.ptr + y*min_disp.pitch);
            disp_row[x].u = float(min_cost_idx)/float(num_disp_layers-1);

            float* min_disp_cost_row = (float*)((char*)min_disp_cost.ptr + y*min_disp_cost.pitch);
            min_disp_cost_row[x] = min_cost;

            /* Search for max cost value */
            float max_cost = -1;
            for(uint32_t z = 0; z < num_disp_layers; z++)
                if(tex3D(cost_volume_tex, x, y, z) > max_cost)
                    max_cost = tex3D(cost_volume_tex, x, y, z);

            float* max_disp_cost_row = (float*)((char*)max_disp_cost.ptr + y*max_disp_cost.pitch);
            max_disp_cost_row[x] = max_cost;

        }
        break;
    default:
        break;
    }

}

/*
 *  Diffusion tensor kernel
 *
 *  Remark: Diffusion tensor is constant so that can be pre-computed and stored/accessed by texture for efficiency.
 *
 */

void __global__ DiffuseTensorKernel(cudaPitchedPtr const diffuse_tensor,
                                    PrimalDualParams const *const pd_params,
                                    CostVolumeParams const *const cv_params,
                                    size_t const width,
                                    size_t const height)
{
    uint32_t const x = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t const y = blockDim.y * blockIdx.y + threadIdx.y;

    uint8_t const max_disp  = cv_params->max_disp;
    uint8_t const win_r     = cv_params->win_r;

    float const alpha   = pd_params->alpha;
    float const beta    = pd_params->beta;

    switch (cv_params->ref_img)
    {
    case LeftRefImage:
        if(x >= max_disp+win_r && x < width-win_r-1 && y >= win_r && y < height-win_r-1)
        {

            DiffuseTensor* diff_tensor_row = (DiffuseTensor*)((char*)diffuse_tensor.ptr + y*diffuse_tensor.pitch);

            float nabla_img[2];
            // original
            nabla_img[0] = tex2D(ref_img_tex, x+1, y) - tex2D(ref_img_tex, x, y);
            nabla_img[1] = tex2D(ref_img_tex, x, y+1) - tex2D(ref_img_tex, x, y);
            //edited
            //nabla_img[0] = tex2D(ref_img_tex, x, y+1) - tex2D(ref_img_tex, x, y);
            //nabla_img[1] = tex2D(ref_img_tex, x+1, y) - tex2D(ref_img_tex, x, y);

            float norm_nabla_img = sqrt(nabla_img[0]*nabla_img[0] + nabla_img[1]*nabla_img[1]);

            float edge_weight = exp(-alpha*pow(norm_nabla_img, beta));

            if(norm_nabla_img > 0)
            {

                float n[2], p_n[2];

                n[0] = nabla_img[0]/norm_nabla_img;
                n[1] = nabla_img[1]/norm_nabla_img;
                p_n[0] = -n[1];
                p_n[1] = n[0];

                /* Diffision tensor D indexed as [0 2; 1 3] */
                diff_tensor_row[x].D[0] = edge_weight*(n[0]*n[0]) + p_n[0]*p_n[0];
                diff_tensor_row[x].D[1] = edge_weight*(n[1]*n[0]) + p_n[1]*p_n[0];
                diff_tensor_row[x].D[2] = edge_weight*(n[0]*n[1]) + p_n[0]*p_n[1];
                diff_tensor_row[x].D[3] = edge_weight*(n[1]*n[1]) + p_n[1]*p_n[1];
            }
            else
            {
                diff_tensor_row[x].D[0] = edge_weight;
                diff_tensor_row[x].D[1] = 0;
                diff_tensor_row[x].D[2] = 0;
                diff_tensor_row[x].D[3] = edge_weight;
            }

        }
        break;
    case RightRefImage:
        if(x >= win_r && x < width-win_r-1 && y >= win_r && y < height-max_disp-win_r-1)
        {

            DiffuseTensor* diff_tensor_row = (DiffuseTensor*)((char*)diffuse_tensor.ptr + y*diffuse_tensor.pitch);

            float nabla_img[2];
            nabla_img[0] = tex2D(ref_img_tex, x+1, y) - tex2D(ref_img_tex, x, y);
            nabla_img[1] = tex2D(ref_img_tex, x, y+1) - tex2D(ref_img_tex, x, y);

            float norm_nabla_img = sqrt(nabla_img[0]*nabla_img[0] + nabla_img[1]*nabla_img[1]);

            float edge_weight = exp(-alpha*pow(norm_nabla_img, beta));

            if(norm_nabla_img > 0)
            {

                float n[2], p_n[2];

                n[0] = nabla_img[0]/norm_nabla_img;
                n[1] = nabla_img[1]/norm_nabla_img;
                p_n[0] = -n[1];
                p_n[1] = n[0];

                /* Diffision tensor D indexed as [0 2; 1 3] */
                diff_tensor_row[x].D[0] = edge_weight*(n[0]*n[0]) + p_n[0]*p_n[0];
                diff_tensor_row[x].D[1] = edge_weight*(n[1]*n[0]) + p_n[1]*p_n[0];
                diff_tensor_row[x].D[2] = edge_weight*(n[0]*n[1]) + p_n[0]*p_n[1];
                diff_tensor_row[x].D[3] = edge_weight*(n[1]*n[1]) + p_n[1]*p_n[1];
            }
            else
            {
                diff_tensor_row[x].D[0] = edge_weight;
                diff_tensor_row[x].D[1] = 0;
                diff_tensor_row[x].D[2] = 0;
                diff_tensor_row[x].D[3] = edge_weight;
            }

        }
        break;
    default:
        break;
    }

}


/*
 *  Diffusion preconditioning kernel
 *
 *  Remark: The step sigma and tau are hard-coded into an image array, so as for the diffusion tensor.
 *
 */

void __global__ DiffusionPrecondKernel(cudaPitchedPtr const primal_step,
                                        cudaPitchedPtr const dual_step,
                                        CostVolumeParams const *const cv_params,
                                        uint32_t const width,
                                        uint32_t const height)
{
    uint32_t const x = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t const y = blockDim.y * blockIdx.y + threadIdx.y;

    uint8_t const max_disp  = cv_params->max_disp;
    uint8_t const win_r     = cv_params->win_r;

    switch (cv_params->ref_img)
    {
    case LeftRefImage:
        if(x >= max_disp+win_r && x < width-win_r && y >= win_r && y < height-win_r)
        {

            PrimalStep* primal_step_row = (PrimalStep*)((char*)primal_step.ptr + y*primal_step.pitch);
            DualStep* dual_step_row = (DualStep*)((char*)dual_step.ptr + y*dual_step.pitch);

            float D[4], Dx[4], Dy[4];

            D[0] = tex2D(diffuse_tensor_tex, x, y).x;
            D[1] = tex2D(diffuse_tensor_tex, x, y).y;
            D[2] = tex2D(diffuse_tensor_tex, x, y).z;
            D[3] = tex2D(diffuse_tensor_tex, x, y).w;

            Dx[0] = tex2D(diffuse_tensor_tex, x-1, y).x;
            Dx[1] = tex2D(diffuse_tensor_tex, x-1, y).y;
            Dx[2] = tex2D(diffuse_tensor_tex, x-1, y).z;
            Dx[3] = tex2D(diffuse_tensor_tex, x-1, y).w;

            Dy[0] = tex2D(diffuse_tensor_tex, x, y-1).x;
            Dy[1] = tex2D(diffuse_tensor_tex, x, y-1).y;
            Dy[2] = tex2D(diffuse_tensor_tex, x, y-1).z;
            Dy[3] = tex2D(diffuse_tensor_tex, x, y-1).w;

            /* D*nabla sum along columns for dual */
            if(x < width-win_r-1)
                dual_step_row[x].sigma[0] = 2*abs(D[0]) + 2*abs(D[2]);

            if(y < height-win_r-1)
                dual_step_row[x].sigma[1] = 2*abs(D[1]) + 2*abs(D[3]);

            dual_step_row[x].sigma[0] = dual_step_row[x].sigma[0] == 0 ? 0:1.0/dual_step_row[x].sigma[0];
            dual_step_row[x].sigma[1] = dual_step_row[x].sigma[1] == 0 ? 0:1.0/dual_step_row[x].sigma[1];

            /* D*nabla sum along rows for primal */
            primal_step_row[x].tau += abs(D[0]) + abs(D[2]);
            primal_step_row[x].tau += abs(D[1]) + abs(D[3]);

            if(x > max_disp+win_r && y > win_r)
                primal_step_row[x].tau += abs(Dx[0]) + abs(Dy[3]);

            if(x == max_disp+win_r && y > win_r)
                primal_step_row[x].tau += abs(Dy[3]);

            if(y == win_r && x > max_disp+win_r)
                primal_step_row[x].tau += abs(Dx[0]);

            primal_step_row[x].tau = primal_step_row[x].tau == 0 ? 0 : 1.0/primal_step_row[x].tau;

        }
        break;
    case RightRefImage:
        if(x >= win_r && x < width-win_r && y >= win_r && y < height-max_disp-win_r)
        {

            PrimalStep* primal_step_row = (PrimalStep*)((char*)primal_step.ptr + y*primal_step.pitch);
            DualStep* dual_step_row = (DualStep*)((char*)dual_step.ptr + y*dual_step.pitch);

            float D[4], Dx[4], Dy[4];

            D[0] = tex2D(diffuse_tensor_tex, x, y).x;
            D[1] = tex2D(diffuse_tensor_tex, x, y).y;
            D[2] = tex2D(diffuse_tensor_tex, x, y).z;
            D[3] = tex2D(diffuse_tensor_tex, x, y).w;

            Dx[0] = tex2D(diffuse_tensor_tex, x-1, y).x;
            Dx[1] = tex2D(diffuse_tensor_tex, x-1, y).y;
            Dx[2] = tex2D(diffuse_tensor_tex, x-1, y).z;
            Dx[3] = tex2D(diffuse_tensor_tex, x-1, y).w;

            Dy[0] = tex2D(diffuse_tensor_tex, x, y-1).x;
            Dy[1] = tex2D(diffuse_tensor_tex, x, y-1).y;
            Dy[2] = tex2D(diffuse_tensor_tex, x, y-1).z;
            Dy[3] = tex2D(diffuse_tensor_tex, x, y-1).w;

            /* D*nabla sum along columns for dual */
            if(x < width-win_r-1)
                dual_step_row[x].sigma[0] = 2*abs(D[0]) + 2*abs(D[2]);

            if(y < height-win_r-1)
                dual_step_row[x].sigma[1] = 2*abs(D[1]) + 2*abs(D[3]);

            dual_step_row[x].sigma[0] = dual_step_row[x].sigma[0] == 0 ? 0:1.0/dual_step_row[x].sigma[0];
            dual_step_row[x].sigma[1] = dual_step_row[x].sigma[1] == 0 ? 0:1.0/dual_step_row[x].sigma[1];

            /* D*nabla sum along rows for primal */
            if(x < width-win_r)
                primal_step_row[x].tau += abs(D[0]) + abs(D[2]);

            if(y < height-max_disp-win_r)
                primal_step_row[x].tau += abs(D[1]) + abs(D[3]);

            if(x >= win_r && y >= win_r)
                primal_step_row[x].tau += abs(Dx[0]) + abs(Dy[3]);

            if(x == win_r && y > win_r)
                primal_step_row[x].tau += abs(Dy[3]);

            if(y == win_r && x > win_r)
                primal_step_row[x].tau += abs(Dx[0]);

            primal_step_row[x].tau = primal_step_row[x].tau == 0 ? 0 : 1.0/primal_step_row[x].tau;

        }
        break;
    default:
        break;
    }

}

/*
 *  Huber-L2 preconditioning dual updating kernel
 *
 *  Remark: Diffusion tensor in float4 texture: [x z; y w].
 *
 */

__global__ void HuberL2DualPrecondKernel(cudaPitchedPtr const dual,
                                         PrimalDualParams const *const pd_params,
                                         CostVolumeParams const *const cv_params,
                                         uint32_t const width,
                                         uint32_t const height)
{

    uint32_t const x = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t const y = blockDim.y * blockIdx.y + threadIdx.y;

    float   const epsilon  = pd_params->epsilon;

    uint8_t const max_disp  = cv_params->max_disp;
    uint8_t const win_r     = cv_params->win_r;

    switch (cv_params->ref_img)
    {
    case LeftRefImage:
        if(x >= max_disp+win_r && x < width-win_r && y >= win_r && y < height-win_r)
        {

            Dual* dual_row = (Dual*)((char*)dual.ptr + y*dual.pitch);

            float D_nabla_primal[2], sigma[2], D[4];

            D[0] = tex2D(diffuse_tensor_tex, x, y).x;
            D[1] = tex2D(diffuse_tensor_tex, x, y).y;
            D[2] = tex2D(diffuse_tensor_tex, x, y).z;
            D[3] = tex2D(diffuse_tensor_tex, x, y).w;

            if(x < width-win_r-1)
                D_nabla_primal[0] = -(D[0]+D[2])*tex2D(head_primal_tex, x, y) +
                                    D[0]*tex2D(head_primal_tex, x+1, y) + D[2]*tex2D(head_primal_tex, x, y+1);

            if(y < height-win_r-1)
                D_nabla_primal[1] = -(D[1]+D[3])*tex2D(head_primal_tex, x, y) +
                                    D[1]*tex2D(head_primal_tex, x+1, y) + D[3]*tex2D(head_primal_tex, x, y+1);

            if(x == width-win_r-1)
                D_nabla_primal[1] = -(D[1]+D[3])*tex2D(head_primal_tex, x, y) + D[3]*tex2D(head_primal_tex, x, y+1);

            if(y == height-win_r-1)
                D_nabla_primal[0] = -(D[0]+D[2])*tex2D(head_primal_tex, x, y) + D[0]*tex2D(head_primal_tex, x+1, y);

            sigma[0] = tex2D(dual_step_tex, x, y).x;
            sigma[1] = tex2D(dual_step_tex, x, y).y;

            for(uint8_t i = 0; i < 2; i++)
                dual_row[x].p[i] = (dual_row[x].p[i] + sigma[i]*D_nabla_primal[i]) / (1.0f + sigma[i]*epsilon);

            // constrain ||p|| <= 1.0
            float p_norm = sqrt(dual_row[x].p[0]*dual_row[x].p[0] + dual_row[x].p[1]*dual_row[x].p[1]);
            float reprojection = fmaxf(1.0f, p_norm);

            for(uint8_t i = 0; i < 2; i++)
                dual_row[x].p[i] /= reprojection;

        }
        break;
    case RightRefImage:
        if(x >= win_r && x < width-win_r && y >= win_r && y < height-max_disp-win_r)
        {

            Dual* dual_row = (Dual*)((char*)dual.ptr + y*dual.pitch);

            float D_nabla_primal[2], sigma[2], D[4];

            D[0] = tex2D(diffuse_tensor_tex, x, y).x;
            D[1] = tex2D(diffuse_tensor_tex, x, y).y;
            D[2] = tex2D(diffuse_tensor_tex, x, y).z;
            D[3] = tex2D(diffuse_tensor_tex, x, y).w;

            if(x < width-win_r-1)
                D_nabla_primal[0] = -(D[0]+D[2])*tex2D(head_primal_tex, x, y) +
                                    D[0]*tex2D(head_primal_tex, x+1, y) + D[2]*tex2D(head_primal_tex, x, y+1);

            if(y < height-max_disp-win_r-1)
                D_nabla_primal[1] = -(D[1]+D[3])*tex2D(head_primal_tex, x, y) +
                                    D[1]*tex2D(head_primal_tex, x+1, y) + D[3]*tex2D(head_primal_tex, x, y+1);

            if(x == width-win_r-1)
                D_nabla_primal[1] = -(D[1]+D[3])*tex2D(head_primal_tex, x, y) + D[3]*tex2D(head_primal_tex, x, y+1);

            if(y == height-max_disp-win_r-1)
                D_nabla_primal[0] = -(D[0]+D[2])*tex2D(head_primal_tex, x, y) + D[0]*tex2D(head_primal_tex, x+1, y);

            sigma[0] = tex2D(dual_step_tex, x, y).x;
            sigma[1] = tex2D(dual_step_tex, x, y).y;

            for(uint8_t i = 0; i < 2; i++)
                dual_row[x].p[i] = (dual_row[x].p[i] + sigma[i]*D_nabla_primal[i]) / (1.0f + sigma[i]*epsilon);

            // constrain ||q|| <= 1.0
            float q_norm = sqrt(dual_row[x].p[0]*dual_row[x].p[0] + dual_row[x].p[1]*dual_row[x].p[1]);
            float reprojection = fmaxf(1.0f, q_norm);

            for(uint8_t i = 0; i < 2; i++)
                dual_row[x].p[i] /= reprojection;

        }
        break;
    default:
        break;
    }

}

/*
 *  Huber-L2 preconditioning primal updating kernel
 *
 *  Remark: Diffusion tensor in float4 texture: [x z; y w].
 *
 */

__global__ void HuberL2PrimalPrecondKernel(cudaPitchedPtr const primal,
                                           PrimalDualParams const *const pd_params,
                                           CostVolumeParams const *const cv_params,
                                           uint32_t const width,
                                           uint32_t const height)
{

    uint32_t const x = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t const y = blockDim.y * blockIdx.y + threadIdx.y;

    float   const aux_theta = pd_params->aux_theta;

    uint8_t const max_disp  = cv_params->max_disp;
    uint8_t const win_r     = cv_params->win_r;

    switch (cv_params->ref_img)
    {
    case LeftRefImage:
        if(x >= max_disp+win_r && x < width-win_r && y >= win_r && y < height-win_r)
        {

            Primal* primal_row = (Primal*)((char*)primal.ptr + y*primal.pitch);

            float div_D_dual = 0.0, D[4], Dx[4], Dy[4];

            D[0] = tex2D(diffuse_tensor_tex, x, y).x;
            D[1] = tex2D(diffuse_tensor_tex, x, y).y;
            D[2] = tex2D(diffuse_tensor_tex, x, y).z;
            D[3] = tex2D(diffuse_tensor_tex, x, y).w;

            Dx[0] = tex2D(diffuse_tensor_tex, x-1, y).x;
            Dx[1] = tex2D(diffuse_tensor_tex, x-1, y).y;
            Dx[2] = tex2D(diffuse_tensor_tex, x-1, y).z;
            Dx[3] = tex2D(diffuse_tensor_tex, x-1, y).w;

            Dy[0] = tex2D(diffuse_tensor_tex, x, y-1).x;
            Dy[1] = tex2D(diffuse_tensor_tex, x, y-1).y;
            Dy[2] = tex2D(diffuse_tensor_tex, x, y-1).z;
            Dy[3] = tex2D(diffuse_tensor_tex, x, y-1).w;

            if(x < width-win_r)
                div_D_dual += -(D[0]+D[2])*tex2D(dual_tex, x, y).x;

            if(y < height-win_r)
                div_D_dual += -(D[1]+D[3])*tex2D(dual_tex, x, y).y;

            if(x > max_disp+win_r)
                div_D_dual += Dx[0]*tex2D(dual_tex, x-1, y).x;

            if(y > win_r)
                div_D_dual += Dy[3]*tex2D(dual_tex, x, y-1).y;

            float tau = tex2D(primal_step_tex, x, y);
            primal_row[x].u = (tex2D(old_primal_tex, x, y) + tau*((1.0/aux_theta)*tex2D(aux_tex, x, y) - div_D_dual)) / (1.0f + tau*(1.0/aux_theta));

        }
        break;
    case RightRefImage:
        if(x >= win_r && x < width-win_r && y >= win_r && y < height-max_disp-win_r)
        {

            Primal* primal_row = (Primal*)((char*)primal.ptr + y*primal.pitch);

            float div_D_dual = 0.0, D[4], Dx[4], Dy[4];

            D[0] = tex2D(diffuse_tensor_tex, x, y).x;
            D[1] = tex2D(diffuse_tensor_tex, x, y).y;
            D[2] = tex2D(diffuse_tensor_tex, x, y).z;
            D[3] = tex2D(diffuse_tensor_tex, x, y).w;

            Dx[0] = tex2D(diffuse_tensor_tex, x-1, y).x;
            Dx[1] = tex2D(diffuse_tensor_tex, x-1, y).y;
            Dx[2] = tex2D(diffuse_tensor_tex, x-1, y).z;
            Dx[3] = tex2D(diffuse_tensor_tex, x-1, y).w;

            Dy[0] = tex2D(diffuse_tensor_tex, x, y-1).x;
            Dy[1] = tex2D(diffuse_tensor_tex, x, y-1).y;
            Dy[2] = tex2D(diffuse_tensor_tex, x, y-1).z;
            Dy[3] = tex2D(diffuse_tensor_tex, x, y-1).w;

            if(x < width-win_r)
                div_D_dual += -(D[0]+D[2])*tex2D(dual_tex, x, y).x;

            if(y < height-max_disp-win_r)
                div_D_dual += -(D[1]+D[3])*tex2D(dual_tex, x, y).y;

            if(x > win_r)
                div_D_dual += Dx[0]*tex2D(dual_tex, x-1, y).x;

            if(y > win_r)
                div_D_dual += Dy[3]*tex2D(dual_tex, x, y-1).y;

            float tau = tex2D(primal_step_tex, x, y);
            primal_row[x].u = (tex2D(old_primal_tex, x, y) + tau*((1.0/aux_theta)*tex2D(aux_tex, x, y) - div_D_dual)) / (1.0f + tau*(1.0/aux_theta));

        }
        break;
    default:
        break;
    }

}

/*
 *  Huber-L2 preconditioning head primal updating kernel
 *
 *  Remark: Diffusion tensor in float4 texture: [x z; y w].
 *
 */

__global__ void HuberL2HeadPrimalKernel(cudaPitchedPtr const head_primal,
                                        PrimalDualParams const *const pd_params,
                                        CostVolumeParams const *const cv_params,
                                        uint32_t const width,
                                        uint32_t const height)
{

    uint32_t const x = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t const y = blockDim.y * blockIdx.y + threadIdx.y;

    float   const theta     = pd_params->theta;

    uint8_t const max_disp  = cv_params->max_disp;
    uint8_t const win_r     = cv_params->win_r;

    switch (cv_params->ref_img)
    {
    case LeftRefImage:
        if(x >= max_disp+win_r && x < width-win_r && y >= win_r && y < height-win_r)
        {

            Primal* head_primal_row = (Primal*)((char*)head_primal.ptr + y*head_primal.pitch);
            head_primal_row[x].u = tex2D(primal_tex, x, y) + theta*(tex2D(primal_tex, x, y) - tex2D(old_primal_tex, x, y));

        }
        break;
    case RightRefImage:
        if(x >= win_r && x < width-win_r && y >= win_r && y < height-max_disp-win_r)
        {

            Primal* head_primal_row = (Primal*)((char*)head_primal.ptr + y*head_primal.pitch);
            head_primal_row[x].u = tex2D(primal_tex, x, y) + theta*(tex2D(primal_tex, x, y) - tex2D(old_primal_tex, x, y));

        }
        break;
    default:
        break;
    }


}

/*
 *  Cost-volume pixel-wise searching kernel
 *
 *  Remark:
 *
 */

__global__ void CostVolumePixelWiseSearch(cudaPitchedPtr const aux,
                                          PrimalDualParams const *const pd_params,
                                          CostVolumeParams const *const cv_params,
                                          uint32_t const width,
                                          uint32_t const height)
{

    uint32_t const x = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t const y = blockDim.y * blockIdx.y + threadIdx.y;

    float const lambda      = pd_params->lambda;
    float const aux_theta   = pd_params->aux_theta;

    uint8_t const max_disp          = cv_params->max_disp;
    uint8_t const win_r             = cv_params->win_r;
    uint8_t const num_disp_layers   = cv_params->num_disp_layers;

    switch (cv_params->ref_img)
    {
    case LeftRefImage:
        if(x >= max_disp+win_r && x < width-win_r && y >= win_r && y < height-win_r)
        {

            Auxiliary* aux_row = (Auxiliary*)((char*)aux.ptr + y*aux.pitch);

            float max_min_cost_diff = tex2D(max_disp_cost_tex, x, y) - tex2D(min_disp_cost_tex, x, y);

            int lower_bound = round((tex2D(primal_tex, x, y) - sqrt(2.0*aux_theta*lambda*max_min_cost_diff))*(cv_params->num_disp_layers-1));
            int upper_bound = round((tex2D(primal_tex, x, y) + sqrt(2.0*aux_theta*lambda*max_min_cost_diff))*(cv_params->num_disp_layers-1));

            lower_bound = lower_bound < 0 ? 0 : lower_bound;
            upper_bound = upper_bound > cv_params->num_disp_layers-1 ? cv_params->num_disp_layers-1 : upper_bound;

            float Eaux_min = 100, Eaux;
            for(uint8_t z = lower_bound; z <= upper_bound; z++)
            {

                float aux_noramlised = (float)z/float(num_disp_layers-1);

                Eaux = 0.5f*(1.0f/aux_theta)*pow((tex2D(primal_tex, x, y) - aux_noramlised),2) + lambda*tex3D(cost_volume_tex, x, y, z);
                if(Eaux < Eaux_min)
                {
                    Eaux_min = Eaux;
                    aux_row[x].a = aux_noramlised;
                }
            }

            /* Sub-sampling using Newton step */
            //        uint8_t z = aux_row[x].a*(num_disp_layers-1);
            //        float nabla_cost;
            //        if(z == 0)
            //            nabla_cost = tex3D(cost_volume_tex, x, y, z+1)-tex3D(cost_volume_tex, x, y, z);
            //        else if(z == num_disp_layers-1)
            //            nabla_cost = tex3D(cost_volume_tex, x, y, z-1)-tex3D(cost_volume_tex, x, y, z);
            //        else
            //            nabla_cost = 0.5*(tex3D(cost_volume_tex, x, y, z+1)-tex3D(cost_volume_tex, x, y, z-1));

            //        aux_row[x].a = tex2D(primal_tex, x, y) + aux_theta*lambda*nabla_cost;

            //        aux_row[x].a = aux_row[x].a > 1.0 ? 1.0 : aux_row[x].a;
            //        aux_row[x].a = aux_row[x].a < 0   ?   0 : aux_row[x].a;

        }
        break;
    case RightRefImage:
        if(x >= win_r && x < width-max_disp-win_r && y >= win_r && y < height-win_r)
        {

            Auxiliary* aux_row = (Auxiliary*)((char*)aux.ptr + y*aux.pitch);

            float max_min_cost_diff = tex2D(max_disp_cost_tex, x, y) - tex2D(min_disp_cost_tex, x, y);

            int lower_bound = round((tex2D(primal_tex, x, y) - sqrt(2.0*aux_theta*lambda*max_min_cost_diff))*(cv_params->num_disp_layers-1));
            int upper_bound = round((tex2D(primal_tex, x, y) + sqrt(2.0*aux_theta*lambda*max_min_cost_diff))*(cv_params->num_disp_layers-1));

            lower_bound = lower_bound < 0 ? 0 : lower_bound;
            upper_bound = upper_bound > cv_params->num_disp_layers-1 ? cv_params->num_disp_layers-1 : upper_bound;

            float Eaux_min = 100, Eaux;
            for(uint8_t z = lower_bound; z <= upper_bound; z++)
            {

                float aux_noramlised = (float)z/float(num_disp_layers-1);

                Eaux = 0.5f*(1.0f/aux_theta)*pow((tex2D(primal_tex, x, y) - aux_noramlised),2) + lambda*tex3D(cost_volume_tex, x, y, z);
                if(Eaux < Eaux_min)
                {
                    Eaux_min = Eaux;
                    aux_row[x].a = aux_noramlised;
                }
            }
        }
        break;
    default:
        break;
    }
}


/*
 *  Huber-L1-Cost-Volume pixel-wise error kernel
 *
 *  Remark: Diffusion tensor in float4 texture: [x z; y w].
 *
 */

__global__ void HuberL1CVErrorKernel(cudaPitchedPtr const error_img,
                                     PrimalDualParams const *const pd_params,
                                     CostVolumeParams const *const cv_params,
                                     uint32_t const width,
                                     uint32_t const height)
{

    uint32_t const x = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t const y = blockDim.y * blockIdx.y + threadIdx.y;

    float const epsilon = pd_params->epsilon;
    float const lambda  = pd_params->lambda;

    uint8_t const max_disp          = cv_params->max_disp;
    uint8_t const win_r             = cv_params->win_r;
    uint8_t const num_disp_layers   = cv_params->num_disp_layers;

    if(x >= max_disp+win_r && x < width-win_r && y >= win_r && y < height-win_r)
    {

        float* error_img_row = (float*)((char*)error_img.ptr + y*error_img.pitch);

        float nabla_primal[2] = {0.0, 0.0};

        if(x < width-win_r-1)
            nabla_primal[0] = tex2D(primal_tex, x+1, y) - tex2D(primal_tex, x, y);

        if(y < height-win_r-1)
            nabla_primal[1] = tex2D(primal_tex, x, y+1) - tex2D(primal_tex, x, y);

        float D_nabla_primal[2] = {0.0, 0.0};

        if(x < width-win_r-1)
            D_nabla_primal[0] = tex2D(diffuse_tensor_tex, x, y).x*nabla_primal[0] + tex2D(diffuse_tensor_tex, x, y).z*nabla_primal[1];

        if(y < height-win_r-1)
            D_nabla_primal[1] = tex2D(diffuse_tensor_tex, x, y).y*nabla_primal[0] + tex2D(diffuse_tensor_tex, x, y).w*nabla_primal[1];

        float huber_norm;
        float norm_reg = sqrt(D_nabla_primal[0]*D_nabla_primal[0]+D_nabla_primal[1]*D_nabla_primal[1]);

        if(norm_reg < epsilon)
            huber_norm = (norm_reg*norm_reg)/(2.0*epsilon);
        else
            abs(D_nabla_primal[0])+abs(D_nabla_primal[1]) - epsilon/2.0;

        uint8_t z = tex2D(primal_tex, x, y)*(num_disp_layers-1);
        float cost_value = tex3D(cost_volume_tex, x, y, z);

        error_img_row[x] = huber_norm + lambda*cost_value;

    }

}

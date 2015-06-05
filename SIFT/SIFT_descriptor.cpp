/**
 * @file SIFT2_descriptor
 * @brief SIFT2 detector + descritpor + BruteForce Matcher + drawing matches with OpenCV functions
 * @author A. Huaman
 */

#include "precomp.hpp"
#include <stdio.h>
#include <iostream>
#include "opencv2/core/core.hpp"
//#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/nonfree/features2d.hpp"
#include <vector>

using namespace cv;

struct temp_storage {
	int i;
	int r;
	int c;
	int o;
};

#define my_abs(x) ((x) >= 0?(x):(-1)*(x))

void readme();
class CV_EXPORTS_W SIFT2 : public Feature2D
{
public:
    CV_WRAP explicit SIFT2( int nfeatures=0, int nOctaveLayers=3,
          double contrastThreshold=0.04, double edgeThreshold=10,
          double sigma=1.6);

    //! returns the descriptor size in floats (128)
    CV_WRAP int descriptorSize() const;

    //! returns the descriptor type
    CV_WRAP int descriptorType() const;

    //! finds the keypoints using SIFT2 algorithm
    void operator()(InputArray img, InputArray mask,
                    vector<KeyPoint>& keypoints) const;
    //! finds the keypoints and computes descriptors for them using SIFT2 algorithm.
    //! Optionally it can compute descriptors for the user-provided keypoints
    void operator()(InputArray img, InputArray mask,
                    vector<KeyPoint>& keypoints,
                    OutputArray descriptors,
                    bool useProvidedKeypoints=false) const;

    void my_operator(InputArray img, InputArray mask,
                    vector<KeyPoint>& keypoints,
                    OutputArray descriptors,
                    bool useProvidedKeypoints=false) const;
    //AlgorithmInfo* info() const;

    void buildGaussianPyramid2( const Mat& base, vector<Mat>& pyr, int nOctaves ) const;
    void buildDoGPyramid2( const vector<Mat>& pyr, vector<Mat>& dogpyr ) const;
    void findScaleSpaceExtrema( const vector<Mat>& gauss_pyr, const vector<Mat>& dog_pyr,
                                vector<KeyPoint>& keypoints ) const;

    void detectImpl( const Mat& image, vector<KeyPoint>& keypoints, const Mat& mask=Mat() ) const;
    void computeImpl( const Mat& image, vector<KeyPoint>& keypoints, Mat& descriptors ) const;

    CV_PROP_RW int nfeatures;
    CV_PROP_RW int nOctaveLayers;
    CV_PROP_RW double contrastThreshold;
    CV_PROP_RW double edgeThreshold;
    CV_PROP_RW double sigma;
};
/**
 * @function main
 * @brief Main function
 */

/******************************* Defs and macros *****************************/

// default number of sampled intervals per octave
static const int SIFT_INTVLS = 3;

// default sigma for initial gaussian smoothing
static const float SIFT_SIGMA = 1.6f;

// default threshold on keypoint contrast |D(x)|
static const float SIFT_CONTR_THR = 0.04f;

// default threshold on keypoint ratio of principle curvatures
static const float SIFT_CURV_THR = 10.f;

// double image size before pyramid construction?
static const bool SIFT_IMG_DBL = true;

// default width of descriptor histogram array
static const int SIFT_DESCR_WIDTH = 4;

// default number of bins per histogram in descriptor array
static const int SIFT_DESCR_HIST_BINS = 8;

// assumed gaussian blur for input image
static const float SIFT_INIT_SIGMA = 0.5f;

// width of border in which to ignore keypoints
static const int SIFT_IMG_BORDER = 5;

// maximum steps of keypoint interpolation before failure
static const int SIFT_MAX_INTERP_STEPS = 5;

// default number of bins in histogram for orientation assignment
static const int SIFT_ORI_HIST_BINS = 36;

// determines gaussian sigma for orientation assignment
static const float SIFT_ORI_SIG_FCTR = 1.5f;

// determines the radius of the region used in orientation assignment
static const float SIFT_ORI_RADIUS = 3 * SIFT_ORI_SIG_FCTR;

// orientation magnitude relative to max that results in new feature
static const float SIFT_ORI_PEAK_RATIO = 0.8f;

// determines the size of a single descriptor orientation histogram
static const float SIFT_DESCR_SCL_FCTR = 3.f;

// threshold on magnitude of elements of descriptor vector
static const float SIFT_DESCR_MAG_THR = 0.2f;

// factor used to convert floating-point descriptor to unsigned char
static const float SIFT_INT_DESCR_FCTR = 512.f;

#if 0
// intermediate type used for DoG pyramids
typedef short sift_wt;
static const int SIFT_FIXPT_SCALE = 48;
#else
// intermediate type used for DoG pyramids
typedef float sift_wt;
static const int SIFT_FIXPT_SCALE = 1;
#endif

static inline void
unpackOctave(const KeyPoint& kpt, int& octave, int& layer, float& scale)
{
    octave = kpt.octave & 255;
    layer = (kpt.octave >> 8) & 255;
    octave = octave < 128 ? octave : (-128 | octave);
    scale = octave >= 0 ? 1.f/(1 << octave) : (float)(1 << -octave);
}

static Mat createInitialImage( const Mat& img, bool doubleImageSize, float sigma )
{
    Mat gray, gray_fpt;
    if( img.channels() == 3 || img.channels() == 4 )
        cvtColor(img, gray, COLOR_BGR2GRAY);
    else
        img.copyTo(gray);
    gray.convertTo(gray_fpt, DataType<sift_wt>::type, SIFT_FIXPT_SCALE, 0);

    float sig_diff;

    if( doubleImageSize )
    {
        sig_diff = sqrtf( std::max(sigma * sigma - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA * 4, 0.01f) );
        Mat dbl;
        resize(gray_fpt, dbl, Size(gray.cols*2, gray.rows*2), 0, 0, INTER_LINEAR);
        GaussianBlur(dbl, dbl, Size(), sig_diff, sig_diff);
        return dbl;
    }
    else
    {
        sig_diff = sqrtf( std::max(sigma * sigma - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA, 0.01f) );
        GaussianBlur(gray_fpt, gray_fpt, Size(), sig_diff, sig_diff);
        return gray_fpt;
    }
}


void SIFT2::buildGaussianPyramid2( const Mat& base, vector<Mat>& pyr, int nOctaves ) const
{
    // nOctaves = 3; nOctaveLayers = 10
    vector<double> sig(nOctaveLayers + 3);
    pyr.resize(nOctaves*(nOctaveLayers + 3)); //Urmish - Thus pyr will now have "reseized" number of elements

    // precompute Gaussian sigmas using the following formula:
    //  \sigma_{total}^2 = \sigma_{i}^2 + \sigma_{i-1}^2
    sig[0] = sigma;
    double k = pow( 2., 1. / nOctaveLayers );
    for( int i = 1; i < nOctaveLayers + 3; i++ )
    {
        double sig_prev = pow(k, (double)(i-1))*sigma;
        double sig_total = sig_prev*k;
        sig[i] = std::sqrt(sig_total*sig_total - sig_prev*sig_prev);
    }

    for( int o = 0; o < nOctaves; o++ ) //3
    {
        for( int i = 0; i < nOctaveLayers + 3; i++ ) //10 //TODO - Could this happen in parallel?
        {
	    //printf("buildGaussianPyramid2 \n");
            Mat& dst = pyr[o*(nOctaveLayers + 3) + i];
            if( o == 0  &&  i == 0 )
                dst = base;
            // base of new octave is halved image from end of previous octave
            else if( i == 0 )
            {
                const Mat& src = pyr[(o-1)*(nOctaveLayers + 3) + nOctaveLayers];
                resize(src, dst, Size(src.cols/2, src.rows/2), 0, 0, INTER_NEAREST); //ACCPOT - This could benefit from SIMD like structure. It is basically an interpolation mechanism
		//TODO - Copy resize here, could use in Aladdin
            }
            else
            {
                const Mat& src = pyr[o*(nOctaveLayers + 3) + i-1];
                GaussianBlur(src, dst, Size(), sig[i], sig[i]); //ACCPOT - Gaussian Blur is averaging. It is convolution over a filter mask. Could accelerate this.
            }
        }
    }
}


void SIFT2::buildDoGPyramid2( const vector<Mat>& gpyr, vector<Mat>& dogpyr ) const
{
    int nOctaves = (int)gpyr.size()/(nOctaveLayers + 3);
    dogpyr.resize( nOctaves*(nOctaveLayers + 2) );

    for( int o = 0; o < nOctaves; o++ ) //3
    {
        for( int i = 0; i < nOctaveLayers + 2; i++ ) //10 TODO - Could this happen in parallel? Assignment to dst is an issue
        {
	    //printf("buildDoGPyramid2 \n");
            const Mat& src1 = gpyr[o*(nOctaveLayers + 3) + i];
            const Mat& src2 = gpyr[o*(nOctaveLayers + 3) + i + 1];
            Mat& dst = dogpyr[o*(nOctaveLayers + 2) + i];
            subtract(src2, src1, dst, noArray(), DataType<sift_wt>::type);
        }
    }
}


// Computes a gradient orientation histogram at a specified pixel
//ACCPOT - This is done for every potential pixel. Could be accelerated
static float calcOrientationHist( const Mat& img, Point pt, int radius,
                                  float sigma, float* hist, int n )
{
    int i, j, k, len = (radius*2+1)*(radius*2+1);

    float expf_scale = -1.f/(2.f * sigma * sigma);
    AutoBuffer<float> buf(len*4 + n+4);
    float *X = buf, *Y = X + len, *Mag = X, *Ori = Y + len, *W = Ori + len;
    float* temphist = W + len + 2;

    for( i = 0; i < n; i++ )
        temphist[i] = 0.f;

    for( i = -radius, k = 0; i <= radius; i++ )
    {
        int y = pt.y + i;
        if( y <= 0 || y >= img.rows - 1 )
            continue;
        for( j = -radius; j <= radius; j++ )
        {
            int x = pt.x + j;
            if( x <= 0 || x >= img.cols - 1 )
                continue;

            float dx = (float)(img.at<sift_wt>(y, x+1) - img.at<sift_wt>(y, x-1));
            float dy = (float)(img.at<sift_wt>(y-1, x) - img.at<sift_wt>(y+1, x));

            X[k] = dx; Y[k] = dy; W[k] = (i*i + j*j)*expf_scale;
            k++;
        }
    }

    len = k;

    // compute gradient values, orientations and the weights over the pixel neighborhood
    exp(W, W, len);
    fastAtan2(Y, X, Ori, len, true);
    magnitude(X, Y, Mag, len);

    for( k = 0; k < len; k++ )
    {
        int bin = cvRound((n/360.f)*Ori[k]);
        if( bin >= n )
            bin -= n;
        if( bin < 0 )
            bin += n;
        temphist[bin] += W[k]*Mag[k];
    }

    // smooth the histogram
    temphist[-1] = temphist[n-1];
    temphist[-2] = temphist[n-2];
    temphist[n] = temphist[0];
    temphist[n+1] = temphist[1];
    for( i = 0; i < n; i++ )
    {
        hist[i] = (temphist[i-2] + temphist[i+2])*(1.f/16.f) +
            (temphist[i-1] + temphist[i+1])*(4.f/16.f) +
            temphist[i]*(6.f/16.f);
    }

    float maxval = hist[0];
    for( i = 1; i < n; i++ )
        maxval = std::max(maxval, hist[i]);

    return maxval;
}


//
// Interpolates a scale-space extremum's location and scale to subpixel
// accuracy to form an image feature. Rejects features with low contrast.
// Based on Section 4 of Lowe's paper.

//TODO - Understand the potential of accelerator here
static bool adjustLocalExtrema( const vector<Mat>& dog_pyr, KeyPoint& kpt, int octv,
                                int& layer, int& r, int& c, int nOctaveLayers,
                                float contrastThreshold, float edgeThreshold, float sigma )
{
    const float img_scale = 1.f/(255*SIFT_FIXPT_SCALE);
    const float deriv_scale = img_scale*0.5f;
    const float second_deriv_scale = img_scale;
    const float cross_deriv_scale = img_scale*0.25f;

    float xi=0, xr=0, xc=0, contr=0;
    int i = 0;

    for( ; i < SIFT_MAX_INTERP_STEPS; i++ )
    {
        int idx = octv*(nOctaveLayers+2) + layer;
        const Mat& img = dog_pyr[idx];
        const Mat& prev = dog_pyr[idx-1];
        const Mat& next = dog_pyr[idx+1];

        Vec3f dD((img.at<sift_wt>(r, c+1) - img.at<sift_wt>(r, c-1))*deriv_scale,
                 (img.at<sift_wt>(r+1, c) - img.at<sift_wt>(r-1, c))*deriv_scale,
                 (next.at<sift_wt>(r, c) - prev.at<sift_wt>(r, c))*deriv_scale);

        float v2 = (float)img.at<sift_wt>(r, c)*2;
        float dxx = (img.at<sift_wt>(r, c+1) + img.at<sift_wt>(r, c-1) - v2)*second_deriv_scale;
        float dyy = (img.at<sift_wt>(r+1, c) + img.at<sift_wt>(r-1, c) - v2)*second_deriv_scale;
        float dss = (next.at<sift_wt>(r, c) + prev.at<sift_wt>(r, c) - v2)*second_deriv_scale;
        float dxy = (img.at<sift_wt>(r+1, c+1) - img.at<sift_wt>(r+1, c-1) -
                     img.at<sift_wt>(r-1, c+1) + img.at<sift_wt>(r-1, c-1))*cross_deriv_scale;
        float dxs = (next.at<sift_wt>(r, c+1) - next.at<sift_wt>(r, c-1) -
                     prev.at<sift_wt>(r, c+1) + prev.at<sift_wt>(r, c-1))*cross_deriv_scale;
        float dys = (next.at<sift_wt>(r+1, c) - next.at<sift_wt>(r-1, c) -
                     prev.at<sift_wt>(r+1, c) + prev.at<sift_wt>(r-1, c))*cross_deriv_scale;

        Matx33f H(dxx, dxy, dxs,
                  dxy, dyy, dys,
                  dxs, dys, dss);

        Vec3f X = H.solve(dD, DECOMP_LU);

        xi = -X[2];
        xr = -X[1];
        xc = -X[0];

        if( std::abs(xi) < 0.5f && std::abs(xr) < 0.5f && std::abs(xc) < 0.5f )
            break;

        if( std::abs(xi) > (float)(INT_MAX/3) ||
            std::abs(xr) > (float)(INT_MAX/3) ||
            std::abs(xc) > (float)(INT_MAX/3) )
            return false;

        c += cvRound(xc);
        r += cvRound(xr);
        layer += cvRound(xi);

        if( layer < 1 || layer > nOctaveLayers ||
            c < SIFT_IMG_BORDER || c >= img.cols - SIFT_IMG_BORDER  ||
            r < SIFT_IMG_BORDER || r >= img.rows - SIFT_IMG_BORDER )
            return false;
    }

    // ensure convergence of interpolation
    if( i >= SIFT_MAX_INTERP_STEPS )
        return false;

    {
        int idx = octv*(nOctaveLayers+2) + layer;
        const Mat& img = dog_pyr[idx];
        const Mat& prev = dog_pyr[idx-1];
        const Mat& next = dog_pyr[idx+1];
        Matx31f dD((img.at<sift_wt>(r, c+1) - img.at<sift_wt>(r, c-1))*deriv_scale,
                   (img.at<sift_wt>(r+1, c) - img.at<sift_wt>(r-1, c))*deriv_scale,
                   (next.at<sift_wt>(r, c) - prev.at<sift_wt>(r, c))*deriv_scale);
        float t = dD.dot(Matx31f(xc, xr, xi));

        contr = img.at<sift_wt>(r, c)*img_scale + t * 0.5f;
        if( std::abs( contr ) * nOctaveLayers < contrastThreshold )
            return false;

        // principal curvatures are computed using the trace and det of Hessian
        float v2 = img.at<sift_wt>(r, c)*2.f;
        float dxx = (img.at<sift_wt>(r, c+1) + img.at<sift_wt>(r, c-1) - v2)*second_deriv_scale;
        float dyy = (img.at<sift_wt>(r+1, c) + img.at<sift_wt>(r-1, c) - v2)*second_deriv_scale;
        float dxy = (img.at<sift_wt>(r+1, c+1) - img.at<sift_wt>(r+1, c-1) -
                     img.at<sift_wt>(r-1, c+1) + img.at<sift_wt>(r-1, c-1)) * cross_deriv_scale;
        float tr = dxx + dyy;
        float det = dxx * dyy - dxy * dxy;

        if( det <= 0 || tr*tr*edgeThreshold >= (edgeThreshold + 1)*(edgeThreshold + 1)*det )
            return false;
    }

    kpt.pt.x = (c + xc) * (1 << octv);
    kpt.pt.y = (r + xr) * (1 << octv);
    kpt.octave = octv + (layer << 8) + (cvRound((xi + 0.5)*255) << 16);
    kpt.size = sigma*powf(2.f, (layer + xi) / nOctaveLayers)*(1 << octv)*2;
    kpt.response = std::abs(contr);

    return true;
}


//
// Detects features at extrema in DoG scale space.  Bad features are discarded
// based on contrast and ratio of principal curvatures.

//void kernel_extrema(float* currptr, float* prevptr, float* nextptr)
void kernel_extrema(uchar* currptr_data, int currptr_int,uchar* prevptr_data, int prevptr_int, uchar* nextptr_data, int nextptr_int, int threshold, int* tempStorageLocation, int& totalPointsOfInterest, int rows, int cols, int step)
{
	//printf("Kernel Extrema\n");
	int totalPointsOfInterest_temp = 0;
	for( int r = SIFT_IMG_BORDER; r < rows-SIFT_IMG_BORDER; r++)
	{
	    //printf("r is %d \n",r);
	    //const sift_wt* currptr = img.ptr<sift_wt>(r); //sift_wt is float
	    const float* currptr = (float *)(currptr_data + currptr_int*r);
	    //const sift_wt* prevptr = prev.ptr<sift_wt>(r);
	    const float* prevptr = (float *)(prevptr_data + prevptr_int*r);
	    //const sift_wt* nextptr = next.ptr<sift_wt>(r);
	    const float* nextptr = (float *)(nextptr_data + nextptr_int*r);
	    //printf("Starting \n");
	    for( int c = SIFT_IMG_BORDER; c < cols-SIFT_IMG_BORDER; c++)
	    {
	        //printf("r,c is %d,%d \n",r,c);
	        float val = currptr[c];
	
	        // find local extrema with pixel accuracy
	        //ACCPOT - Takes up  lot of time. Could this be accelerated?
	        if( my_abs(val) > threshold &&
	           ((val > 0 && val >= currptr[c-1] && val >= currptr[c+1] &&
	             val >= currptr[c-step-1] && val >= currptr[c-step] && val >= currptr[c-step+1] &&
	             val >= currptr[c+step-1] && val >= currptr[c+step] && val >= currptr[c+step+1] &&
	             val >= nextptr[c] && val >= nextptr[c-1] && val >= nextptr[c+1] &&
	             val >= nextptr[c-step-1] && val >= nextptr[c-step] && val >= nextptr[c-step+1] &&
	             val >= nextptr[c+step-1] && val >= nextptr[c+step] && val >= nextptr[c+step+1] &&
	             val >= prevptr[c] && val >= prevptr[c-1] && val >= prevptr[c+1] &&
	             val >= prevptr[c-step-1] && val >= prevptr[c-step] && val >= prevptr[c-step+1] &&
	             val >= prevptr[c+step-1] && val >= prevptr[c+step] && val >= prevptr[c+step+1]) ||
	            (val < 0 && val <= currptr[c-1] && val <= currptr[c+1] &&
	             val <= currptr[c-step-1] && val <= currptr[c-step] && val <= currptr[c-step+1] &&
	             val <= currptr[c+step-1] && val <= currptr[c+step] && val <= currptr[c+step+1] &&
	             val <= nextptr[c] && val <= nextptr[c-1] && val <= nextptr[c+1] &&
	             val <= nextptr[c-step-1] && val <= nextptr[c-step] && val <= nextptr[c-step+1] &&
	             val <= nextptr[c+step-1] && val <= nextptr[c+step] && val <= nextptr[c+step+1] &&
	             val <= prevptr[c] && val <= prevptr[c-1] && val <= prevptr[c+1] &&
	             val <= prevptr[c-step-1] && val <= prevptr[c-step] && val <= prevptr[c-step+1] &&
	             val <= prevptr[c+step-1] && val <= prevptr[c+step] && val <= prevptr[c+step+1])))
	        {
	    	//Need r,c,i,o
	            //int r1 = r, c1 = c, layer = i;
			tempStorageLocation[totalPointsOfInterest_temp] = r;
			tempStorageLocation[totalPointsOfInterest_temp+1] = c;
			totalPointsOfInterest_temp = totalPointsOfInterest_temp + 2;
	    	//temp_storage temp_storage_obj = {i,r,c,o};
	    	//potential_keypoints.push_back(temp_storage_obj);
	            
	        }
	    }
	    //printf("Over %d\n",totalPointsOfInterest_temp);
	}
	totalPointsOfInterest = totalPointsOfInterest_temp;
	//printf("Points of interest are %d\n",totalPointsOfInterest_temp);
}

void SIFT2::findScaleSpaceExtrema( const vector<Mat>& gauss_pyr, const vector<Mat>& dog_pyr,
                                  vector<KeyPoint>& keypoints ) const
{
    int nOctaves = (int)gauss_pyr.size()/(nOctaveLayers + 3);
    int threshold = cvFloor(0.5 * contrastThreshold / nOctaveLayers * 255 * SIFT_FIXPT_SCALE);
    const int n = SIFT_ORI_HIST_BINS;
    float hist[n];
    KeyPoint kpt;

    keypoints.clear();
    int count=1;
    //printf("nOctaves=%d, nOctaveLayers=%d\n",nOctaves,nOctaveLayers);
    for( int o = 0; o < nOctaves; o++ ) //10
        for( int i = 1; i <= nOctaveLayers; i++ ) //3 TODO - Potential to parallely execute this? 
						  //ACCPOT - Yes, bin calculated individually and then merged	
        {
	    //printf("Count - %d\n",count);
	    count++;
            int idx = o*(nOctaveLayers+2)+i;
            const Mat& img = dog_pyr[idx];
            const Mat& prev = dog_pyr[idx-1];
            const Mat& next = dog_pyr[idx+1];
            int step = (int)img.step1();
            int rows = img.rows, cols = img.cols;
	    uchar* currptr_data = img.data;
	    int currptr_int = img.step.p[0];		

	    uchar* prevptr_data = prev.data;
	    int prevptr_int = prev.step.p[0];		

	    uchar* nextptr_data = next.data;
	    int nextptr_int = next.step.p[0];		
	    int sizeOfPoints = 0;
	    int* tempStorageLocation = (int *)malloc(sizeof(int)*cols*rows);
	    //printf("roes,cols = %d,%d",rows,cols);
	    kernel_extrema(currptr_data,currptr_int, prevptr_data, prevptr_int, nextptr_data, nextptr_int, threshold, tempStorageLocation, sizeOfPoints, rows, cols, step);
	    //printf("Return from kernel call with %d\n",sizeOfPoints);
	    for (int iterator = 0; iterator < sizeOfPoints; iterator=iterator+2)
	    {
		//printf("Iterator %d\n",iterator);

		int layer = i;
		int r1 = tempStorageLocation[iterator];
		int c1 = tempStorageLocation[iterator+1];
		if( !adjustLocalExtrema(dog_pyr, kpt, o, layer, r1, c1, nOctaveLayers, (float)contrastThreshold, (float)edgeThreshold, (float)sigma) )
                    continue;
                float scl_octv = kpt.size*0.5f/(1 << o);
                float omax = calcOrientationHist(gauss_pyr[o*(nOctaveLayers+3) + layer],
                                                 Point(c1, r1),
                                                 cvRound(SIFT_ORI_RADIUS * scl_octv),
                                                 SIFT_ORI_SIG_FCTR * scl_octv,
                                                 hist, n);
                float mag_thr = (float)(omax * SIFT_ORI_PEAK_RATIO);
                for( int j = 0; j < n; j++ )
                {
                    int l = j > 0 ? j - 1 : n - 1;
                    int r2 = j < n-1 ? j + 1 : 0;

                    if( hist[j] > hist[l]  &&  hist[j] > hist[r2]  &&  hist[j] >= mag_thr )
                    {
                        float bin = j + 0.5f * (hist[l]-hist[r2]) / (hist[l] - 2*hist[j] + hist[r2]);
                        bin = bin < 0 ? n + bin : bin >= n ? bin - n : bin;
                        kpt.angle = 360.f - (float)((360.f/n) * bin);
                        if(std::abs(kpt.angle - 360.f) < FLT_EPSILON)
                            kpt.angle = 0.f;
                        keypoints.push_back(kpt);
                    }
                }
	     }

        }
	//printf("Potential Keypoints size is %d\n",potential_keypoints.size());
	//Iterate here
	

	//Add code for calcOrientationHist and adjustLocalExtremaHere
}


static void calcSIFT2Descriptor( const Mat& img, Point2f ptf, float ori, float scl,
                               int d, int n, float* dst )
{
    Point pt(cvRound(ptf.x), cvRound(ptf.y));
    float cos_t = cosf(ori*(float)(CV_PI/180)); //ACCPOT - TRIG FUNCTION
    float sin_t = sinf(ori*(float)(CV_PI/180)); //ACCPOT - TRIG FUNCTION
    float bins_per_rad = n / 360.f;
    float exp_scale = -1.f/(d * d * 0.5f);
    float hist_width = SIFT_DESCR_SCL_FCTR * scl;
    int radius = cvRound(hist_width * 1.4142135623730951f * (d + 1) * 0.5f);
    // Clip the radius to the diagonal of the image to avoid autobuffer too large exception
    radius = std::min(radius, (int) sqrt((double) img.cols*img.cols + img.rows*img.rows));
    cos_t /= hist_width;
    sin_t /= hist_width;

    int i, j, k, len = (radius*2+1)*(radius*2+1), histlen = (d+2)*(d+2)*(n+2);
    int rows = img.rows, cols = img.cols;

    AutoBuffer<float> buf(len*6 + histlen);
    float *X = buf, *Y = X + len, *Mag = Y, *Ori = Mag + len, *W = Ori + len;
    float *RBin = W + len, *CBin = RBin + len, *hist = CBin + len;

    for( i = 0; i < d+2; i++ )
    {
        for( j = 0; j < d+2; j++ )
            for( k = 0; k < n+2; k++ )
                hist[(i*(d+2) + j)*(n+2) + k] = 0.;
    }
//ACCPOT - Below loop seems to execute in parallel
    for( i = -radius, k = 0; i <= radius; i++ )
        for( j = -radius; j <= radius; j++ )
        {
            // Calculate sample's histogram array coords rotated relative to ori.
            // Subtract 0.5 so samples that fall e.g. in the center of row 1 (i.e.
            // r_rot = 1.5) have full weight placed in row 1 after interpolation.
            float c_rot = j * cos_t - i * sin_t;
            float r_rot = j * sin_t + i * cos_t;
            float rbin = r_rot + d/2 - 0.5f;
            float cbin = c_rot + d/2 - 0.5f;
            int r = pt.y + i, c = pt.x + j;

            if( rbin > -1 && rbin < d && cbin > -1 && cbin < d &&
                r > 0 && r < rows - 1 && c > 0 && c < cols - 1 )
            {
                float dx = (float)(img.at<sift_wt>(r, c+1) - img.at<sift_wt>(r, c-1));
                float dy = (float)(img.at<sift_wt>(r-1, c) - img.at<sift_wt>(r+1, c));
                X[k] = dx; Y[k] = dy; RBin[k] = rbin; CBin[k] = cbin;
                W[k] = (c_rot * c_rot + r_rot * r_rot)*exp_scale;
                k++;
            }
        }

    len = k;
    fastAtan2(Y, X, Ori, len, true); //ACCPOT - TRIG Function
    magnitude(X, Y, Mag, len);
    exp(W, W, len);
//ACCPOT - This loop can be unrolled
    for( k = 0; k < len; k++ )
    {
        float rbin = RBin[k], cbin = CBin[k];
        float obin = (Ori[k] - ori)*bins_per_rad;
        float mag = Mag[k]*W[k];

        int r0 = cvFloor( rbin );
        int c0 = cvFloor( cbin );
        int o0 = cvFloor( obin );
        rbin -= r0;
        cbin -= c0;
        obin -= o0;

        if( o0 < 0 )
            o0 += n;
        if( o0 >= n )
            o0 -= n;

        // histogram update using tri-linear interpolation
        float v_r1 = mag*rbin, v_r0 = mag - v_r1;
        float v_rc11 = v_r1*cbin, v_rc10 = v_r1 - v_rc11;
        float v_rc01 = v_r0*cbin, v_rc00 = v_r0 - v_rc01;
        float v_rco111 = v_rc11*obin, v_rco110 = v_rc11 - v_rco111;
        float v_rco101 = v_rc10*obin, v_rco100 = v_rc10 - v_rco101;
        float v_rco011 = v_rc01*obin, v_rco010 = v_rc01 - v_rco011;
        float v_rco001 = v_rc00*obin, v_rco000 = v_rc00 - v_rco001;

        int idx = ((r0+1)*(d+2) + c0+1)*(n+2) + o0;
        hist[idx] += v_rco000;
        hist[idx+1] += v_rco001;
        hist[idx+(n+2)] += v_rco010;
        hist[idx+(n+3)] += v_rco011;
        hist[idx+(d+2)*(n+2)] += v_rco100;
        hist[idx+(d+2)*(n+2)+1] += v_rco101;
        hist[idx+(d+3)*(n+2)] += v_rco110;
        hist[idx+(d+3)*(n+2)+1] += v_rco111;
    }

    // finalize histogram, since the orientation histograms are circular
    for( i = 0; i < d; i++ )
        for( j = 0; j < d; j++ )
        {
            int idx = ((i+1)*(d+2) + (j+1))*(n+2);
            hist[idx] += hist[idx+n];
            hist[idx+1] += hist[idx+n+1];
            for( k = 0; k < n; k++ )
                dst[(i*d + j)*n + k] = hist[idx+k];
        }
    // copy histogram to the descriptor,
    // apply hysteresis thresholding
    // and scale the result, so that it can be easily converted
    // to byte array
    float nrm2 = 0;
    len = d*d*n;
    for( k = 0; k < len; k++ )
        nrm2 += dst[k]*dst[k];
    float thr = std::sqrt(nrm2)*SIFT_DESCR_MAG_THR;
    for( i = 0, nrm2 = 0; i < k; i++ )
    {
        float val = std::min(dst[i], thr);
        dst[i] = val;
        nrm2 += val*val;
    }
    nrm2 = SIFT_INT_DESCR_FCTR/std::max(std::sqrt(nrm2), FLT_EPSILON);

#if 1
    for( k = 0; k < len; k++ )
    {
        dst[k] = saturate_cast<uchar>(dst[k]*nrm2);
    }
#else
    float nrm1 = 0;
    for( k = 0; k < len; k++ )
    {
        dst[k] *= nrm2;
        nrm1 += dst[k];
    }
    nrm1 = 1.f/std::max(nrm1, FLT_EPSILON);
    for( k = 0; k < len; k++ )
    {
        dst[k] = std::sqrt(dst[k] * nrm1);//saturate_cast<uchar>(std::sqrt(dst[k] * nrm1)*SIFT_INT_DESCR_FCTR);
    }
#endif
}

static void calcDescriptors(const vector<Mat>& gpyr, const vector<KeyPoint>& keypoints,
                            Mat& descriptors, int nOctaveLayers, int firstOctave )
{
    int d = SIFT_DESCR_WIDTH, n = SIFT_DESCR_HIST_BINS;

    for( size_t i = 0; i < keypoints.size(); i++ )
    {
        KeyPoint kpt = keypoints[i];
        int octave, layer;
        float scale;
        unpackOctave(kpt, octave, layer, scale);
        CV_Assert(octave >= firstOctave && layer <= nOctaveLayers+2);
        float size=kpt.size*scale;
        Point2f ptf(kpt.pt.x*scale, kpt.pt.y*scale);
        const Mat& img = gpyr[(octave - firstOctave)*(nOctaveLayers + 3) + layer];

        float angle = 360.f - kpt.angle;
        if(std::abs(angle - 360.f) < FLT_EPSILON)
            angle = 0.f;
        calcSIFT2Descriptor(img, ptf, angle, size*0.5f, d, n, descriptors.ptr<float>((int)i));
    }
}

//////////////////////////////////////////////////////////////////////////////////////////

SIFT2::SIFT2( int _nfeatures, int _nOctaveLayers,
           double _contrastThreshold, double _edgeThreshold, double _sigma )
    : nfeatures(_nfeatures), nOctaveLayers(_nOctaveLayers),
    contrastThreshold(_contrastThreshold), edgeThreshold(_edgeThreshold), sigma(_sigma)
{
}

int SIFT2::descriptorSize() const
{
    return SIFT_DESCR_WIDTH*SIFT_DESCR_WIDTH*SIFT_DESCR_HIST_BINS;
}

int SIFT2::descriptorType() const
{
    return CV_32F;
}


void SIFT2::operator()(InputArray _image, InputArray _mask,
                      vector<KeyPoint>& keypoints) const
{
//    (*this)(_image, _mask, keypoints, noArray());
    (*this).my_operator(_image, _mask, keypoints, noArray());
}


void SIFT2::operator()(InputArray _image, InputArray _mask,
                      vector<KeyPoint>& keypoints,
                      OutputArray _descriptors,
                      bool useProvidedKeypoints) const
{
	std::cout<<"This should not be used!!!! "<<std::endl;
}
void SIFT2::my_operator(InputArray _image, InputArray _mask,
                      vector<KeyPoint>& keypoints,
                      OutputArray _descriptors,
                      bool useProvidedKeypoints) const
{
    //std::cout<<"Using my_operator "<<std::endl;
    int firstOctave = -1, actualNOctaves = 0, actualNLayers = 0;
    Mat image = _image.getMat(), mask = _mask.getMat();

    if( image.empty() || image.depth() != CV_8U )
        CV_Error( CV_StsBadArg, "image is empty or has incorrect depth (!=CV_8U)" );

    if( !mask.empty() && mask.type() != CV_8UC1 )
        CV_Error( CV_StsBadArg, "mask has incorrect type (!=CV_8UC1)" );

    if( useProvidedKeypoints )
    {
        firstOctave = 0;
        int maxOctave = INT_MIN;
        for( size_t i = 0; i < keypoints.size(); i++ )
        {
            int octave, layer;
            float scale;
            unpackOctave(keypoints[i], octave, layer, scale);
            firstOctave = std::min(firstOctave, octave);
            maxOctave = std::max(maxOctave, octave);
            actualNLayers = std::max(actualNLayers, layer-2);
        }

        firstOctave = std::min(firstOctave, 0);
        CV_Assert( firstOctave >= -1 && actualNLayers <= nOctaveLayers );
        actualNOctaves = maxOctave - firstOctave + 1;
    }

    Mat base = createInitialImage(image, firstOctave < 0, (float)sigma);
    vector<Mat> gpyr, dogpyr;
    int nOctaves = actualNOctaves > 0 ? actualNOctaves : cvRound(log( (double)std::min( base.cols, base.rows ) ) / log(2.) - 2) - firstOctave;

    //double t, tf = getTickFrequency();
    //t = (double)getTickCount();
    buildGaussianPyramid2(base, gpyr, nOctaves);
    buildDoGPyramid2(gpyr, dogpyr);

    //t = (double)getTickCount() - t;
    //printf("pyramid construction time: %g\n", t*1000./tf);

    if( !useProvidedKeypoints )
    {
        //t = (double)getTickCount();
        findScaleSpaceExtrema(gpyr, dogpyr, keypoints);
        KeyPointsFilter::removeDuplicated( keypoints );

        if( nfeatures > 0 )
            KeyPointsFilter::retainBest(keypoints, nfeatures);
        //t = (double)getTickCount() - t;
        //printf("keypoint detection time: %g\n", t*1000./tf);

        if( firstOctave < 0 )
            for( size_t i = 0; i < keypoints.size(); i++ )
            {
                KeyPoint& kpt = keypoints[i];
                float scale = 1.f/(float)(1 << -firstOctave);
                kpt.octave = (kpt.octave & ~255) | ((kpt.octave + firstOctave) & 255);
                kpt.pt *= scale;
                kpt.size *= scale;
            }

        if( !mask.empty() )
            KeyPointsFilter::runByPixelsMask( keypoints, mask );
    }
    else
    {
        // filter keypoints by mask
        //KeyPointsFilter::runByPixelsMask( keypoints, mask );
    }

    if( _descriptors.needed() )
    {
        //t = (double)getTickCount();
        int dsize = descriptorSize();
        _descriptors.create((int)keypoints.size(), dsize, CV_32F);
        Mat descriptors = _descriptors.getMat();

        calcDescriptors(gpyr, keypoints, descriptors, nOctaveLayers, firstOctave);
        //t = (double)getTickCount() - t;
        //printf("descriptor extraction time: %g\n", t*1000./tf);
    }
}

void SIFT2::detectImpl( const Mat& image, vector<KeyPoint>& keypoints, const Mat& mask) const
{
    std::cout <<"detectImpl: New Implementation of SIFT "<< std::endl;
    //(*this)(image, mask, keypoints, noArray());
    (*this).my_operator(image, mask, keypoints, noArray());
}

void SIFT2::computeImpl( const Mat& image, vector<KeyPoint>& keypoints, Mat& descriptors) const
{
    std::cout <<"computeImpl: New Implementation of SIFT "<< std::endl;
    //(*this)(image, Mat(), keypoints, descriptors, true);
    (*this).my_operator(image, Mat(), keypoints, descriptors, true);
}
int main( int argc, char** argv )
{
  if( argc != 3 )
  { return -1; }

  Mat img_1 = imread( argv[1], CV_LOAD_IMAGE_GRAYSCALE );
  Mat img_2 = imread( argv[2], CV_LOAD_IMAGE_GRAYSCALE );

  if( !img_1.data || !img_2.data )
  { return -1; }

  //-- Step 1: Detect the keypoints using SIFT Detector
  int minHessian = 400;

  SIFT2 detector( minHessian );

  std::vector<KeyPoint> keypoints_1, keypoints_2;

  detector.detectImpl( img_1, keypoints_1 );
  printf("Keypoints_1 size is - %d\n",keypoints_1.size());
  detector.detectImpl( img_2, keypoints_2 );
  printf("Keypoints_2 size is - %d\n",keypoints_2.size());

  //-- Step 2: Calculate descriptors (feature vectors)
  SIFT2 extractor;

  Mat descriptors_1, descriptors_2;

  extractor.computeImpl( img_1, keypoints_1, descriptors_1 );
  extractor.computeImpl( img_2, keypoints_2, descriptors_2 );

  //-- Step 3: Matching descriptor vectors with a brute force matcher
  BFMatcher matcher(NORM_L2);
  std::vector< DMatch > matches;
  matcher.match( descriptors_1, descriptors_2, matches );

  //-- Draw matches
  Mat img_matches;
  drawMatches( img_1, keypoints_1, img_2, keypoints_2, matches, img_matches );

  //-- Show detected matches
  imshow("Matches", img_matches );

  waitKey(0);

  return 0;
}

/**
 * @function readme
 */
void readme()
{ std::cout << " Usage: ./SIFT_descriptor <img1> <img2>" << std::endl; }

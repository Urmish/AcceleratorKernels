/*
 * matching_test.cpp
 *
 *  Created on: Oct 17, 2010
 *      Author: ethan
 */
#include "precomp.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/calib3d/calib3d.hpp"
//#include "opencv2/features2d/features2d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <vector>
#include <iostream>

using namespace cv;
using namespace std;
class CV_EXPORTS_W My_FastFeatureDetector : public FeatureDetector
{
public:

    enum
    { // Define it in old class to simplify migration to 2.5
      TYPE_5_8 = 0, TYPE_7_12 = 1, TYPE_9_16 = 2
    };

    CV_WRAP My_FastFeatureDetector( int threshold=10, bool nonmaxSuppression=true );

    virtual void detectImpl( const Mat& image, vector<KeyPoint>& keypoints, const Mat& mask=Mat() ) const;

    int threshold;
    bool nonmaxSuppression;
};


void makeOffsets(int pixel[25], int row_stride, int patternSize);

template<int patternSize>
int cornerScore(const uchar* ptr, const int pixel[], int threshold);

void makeOffsets(int pixel[25], int rowStride, int patternSize)
{
    static const int offsets16[][2] =
    {
        {0,  3}, { 1,  3}, { 2,  2}, { 3,  1}, { 3, 0}, { 3, -1}, { 2, -2}, { 1, -3},
        {0, -3}, {-1, -3}, {-2, -2}, {-3, -1}, {-3, 0}, {-3,  1}, {-2,  2}, {-1,  3}
    };

    static const int offsets12[][2] =
    {
        {0,  2}, { 1,  2}, { 2,  1}, { 2, 0}, { 2, -1}, { 1, -2},
        {0, -2}, {-1, -2}, {-2, -1}, {-2, 0}, {-2,  1}, {-1,  2}
    };

    static const int offsets8[][2] =
    {
        {0,  1}, { 1,  1}, { 1, 0}, { 1, -1},
        {0, -1}, {-1, -1}, {-1, 0}, {-1,  1}
    };

    const int (*offsets)[2] = patternSize == 16 ? offsets16 :
                              patternSize == 12 ? offsets12 :
                              patternSize == 8  ? offsets8  : 0;

    CV_Assert(pixel && offsets);

    int k = 0;
    for( ; k < patternSize; k++ )
        pixel[k] = offsets[k][0] + offsets[k][1] * rowStride;
    for( ; k < 25; k++ )
        pixel[k] = pixel[k - patternSize];
}


template<>
int cornerScore<16>(const uchar* ptr, const int pixel[], int threshold)
{
    const int K = 8, N = K*3 + 1;
    int k, v = ptr[0];
    short d[N];
    for( k = 0; k < N; k++ )
        d[k] = (short)(v - ptr[pixel[k]]);

    int a0 = threshold;
    for( k = 0; k < 16; k += 2 )
    {
        int a = std::min((int)d[k+1], (int)d[k+2]);
        a = std::min(a, (int)d[k+3]);
        if( a <= a0 )
            continue;
        a = std::min(a, (int)d[k+4]);
        a = std::min(a, (int)d[k+5]);
        a = std::min(a, (int)d[k+6]);
        a = std::min(a, (int)d[k+7]);
        a = std::min(a, (int)d[k+8]);
        a0 = std::max(a0, std::min(a, (int)d[k]));
        a0 = std::max(a0, std::min(a, (int)d[k+9]));
    }

    int b0 = -a0;
    for( k = 0; k < 16; k += 2 )
    {
        int b = std::max((int)d[k+1], (int)d[k+2]);
        b = std::max(b, (int)d[k+3]);
        b = std::max(b, (int)d[k+4]);
        b = std::max(b, (int)d[k+5]);
        if( b >= b0 )
            continue;
        b = std::max(b, (int)d[k+6]);
        b = std::max(b, (int)d[k+7]);
        b = std::max(b, (int)d[k+8]);

        b0 = std::min(b0, std::max(b, (int)d[k]));
        b0 = std::min(b0, std::max(b, (int)d[k+9]));
    }

    threshold = -b0-1;
    return threshold;
}

template<>
int cornerScore<12>(const uchar* ptr, const int pixel[], int threshold)
{
    const int K = 6, N = K*3 + 1;
    int k, v = ptr[0];
    short d[N + 4];
    for( k = 0; k < N; k++ )
        d[k] = (short)(v - ptr[pixel[k]]);
    int a0 = threshold;
    for( k = 0; k < 12; k += 2 )
    {
        int a = std::min((int)d[k+1], (int)d[k+2]);
        if( a <= a0 )
            continue;
        a = std::min(a, (int)d[k+3]);
        a = std::min(a, (int)d[k+4]);
        a = std::min(a, (int)d[k+5]);
        a = std::min(a, (int)d[k+6]);
        a0 = std::max(a0, std::min(a, (int)d[k]));
        a0 = std::max(a0, std::min(a, (int)d[k+7]));
    }

    int b0 = -a0;
    for( k = 0; k < 12; k += 2 )
    {
        int b = std::max((int)d[k+1], (int)d[k+2]);
        b = std::max(b, (int)d[k+3]);
        b = std::max(b, (int)d[k+4]);
        if( b >= b0 )
            continue;
        b = std::max(b, (int)d[k+5]);
        b = std::max(b, (int)d[k+6]);

        b0 = std::min(b0, std::max(b, (int)d[k]));
        b0 = std::min(b0, std::max(b, (int)d[k+7]));
    }

    threshold = -b0-1;
    return threshold;
}

template<>
int cornerScore<8>(const uchar* ptr, const int pixel[], int threshold)
{
    const int K = 4, N = K*3 + 1;
    int k, v = ptr[0];
    short d[N];
    for( k = 0; k < N; k++ )
        d[k] = (short)(v - ptr[pixel[k]]);

    int a0 = threshold;
    for( k = 0; k < 8; k += 2 )
    {
        int a = std::min((int)d[k+1], (int)d[k+2]);
        if( a <= a0 )
            continue;
        a = std::min(a, (int)d[k+3]);
        a = std::min(a, (int)d[k+4]);
        a0 = std::max(a0, std::min(a, (int)d[k]));
        a0 = std::max(a0, std::min(a, (int)d[k+5]));
    }

    int b0 = -a0;
    for( k = 0; k < 8; k += 2 )
    {
        int b = std::max((int)d[k+1], (int)d[k+2]);
        b = std::max(b, (int)d[k+3]);
        if( b >= b0 )
            continue;
        b = std::max(b, (int)d[k+4]);

        b0 = std::min(b0, std::max(b, (int)d[k]));
        b0 = std::min(b0, std::max(b, (int)d[k+5]));
    }

    threshold = -b0-1;

   return threshold;
}

template<int patternSize>
void FAST_t(InputArray _img, std::vector<KeyPoint>& keypoints, int threshold, bool nonmax_suppression)
{
    Mat img = _img.getMat(); //Pass this as an argument
    const int K = patternSize/2, N = patternSize + K + 1;
    int i, j, k, pixel[25];
    makeOffsets(pixel, (int)img.step, patternSize); //Make this function inline if possible

    keypoints.clear(); //Do this outside the function

    threshold = std::min(std::max(threshold, 0), 255); //Again, this is something that could be done outside the function call, but where is threshold????

    uchar threshold_tab[512];
    for( i = -255; i <= 255; i++ )
        threshold_tab[i+255] = (uchar)(i < -threshold ? 1 : i > threshold ? 2 : 0);

    AutoBuffer<uchar> _buf((img.cols+16)*3*(sizeof(int) + sizeof(uchar)) + 128); //This could be passed on as an argument
    uchar* buf[3];
    buf[0] = _buf; buf[1] = buf[0] + img.cols; buf[2] = buf[1] + img.cols;
    int* cpbuf[3];
    cpbuf[0] = (int*)alignPtr(buf[2] + img.cols, sizeof(int)) + 1; //Need to pass img.cols and img.rows
    cpbuf[1] = cpbuf[0] + img.cols + 1;
    cpbuf[2] = cpbuf[1] + img.cols + 1;
    memset(buf[0], 0, img.cols*3);

    for(i = 3; i < img.rows-2; i++)
    {
        const uchar* ptr = img.ptr<uchar>(i) + 3;
        uchar* curr = buf[(i - 3)%3];
        int* cornerpos = cpbuf[(i - 3)%3];
        memset(curr, 0, img.cols);
        int ncorners = 0;

        if( i < img.rows - 3 )
        {
            j = 3;
            for( ; j < img.cols - 3; j++, ptr++ )
            {
                int v = ptr[0];
                const uchar* tab = &threshold_tab[0] - v + 255;
                int d = tab[ptr[pixel[0]]] | tab[ptr[pixel[8]]];

                if( d == 0 )
                    continue;

                d &= tab[ptr[pixel[2]]] | tab[ptr[pixel[10]]];
                d &= tab[ptr[pixel[4]]] | tab[ptr[pixel[12]]];
                d &= tab[ptr[pixel[6]]] | tab[ptr[pixel[14]]];

                if( d == 0 )
                    continue;

                d &= tab[ptr[pixel[1]]] | tab[ptr[pixel[9]]];
                d &= tab[ptr[pixel[3]]] | tab[ptr[pixel[11]]];
                d &= tab[ptr[pixel[5]]] | tab[ptr[pixel[13]]];
                d &= tab[ptr[pixel[7]]] | tab[ptr[pixel[15]]];

                if( d & 1 )
                {
                    int vt = v - threshold, count = 0;

                    for( k = 0; k < N; k++ )
                    {
                        int x = ptr[pixel[k]];
                        if(x < vt)
                        {
                            if( ++count > K )
                            {
                                cornerpos[ncorners++] = j;
                                if(nonmax_suppression)
                                    curr[j] = (uchar)cornerScore<patternSize>(ptr, pixel, threshold);
                                break;
                            }
                        }
                        else
                            count = 0;
                    }
                }

                if( d & 2 )
                {
                    int vt = v + threshold, count = 0;

                    for( k = 0; k < N; k++ )
                    {
                        int x = ptr[pixel[k]];
                        if(x > vt)
                        {
                            if( ++count > K )
                            {
                                cornerpos[ncorners++] = j;
                                if(nonmax_suppression)
                                    curr[j] = (uchar)cornerScore<patternSize>(ptr, pixel, threshold);
                                break;
                            }
                        }
                        else
                            count = 0;
                    }
                }
            }
        }

        cornerpos[-1] = ncorners;

        if( i == 3 )
            continue;

        const uchar* prev = buf[(i - 4 + 3)%3];
        const uchar* pprev = buf[(i - 5 + 3)%3];
        cornerpos = cpbuf[(i - 4 + 3)%3];
        ncorners = cornerpos[-1];
	//ACCPOT - This could be accelerated in hardware
        for( k = 0; k < ncorners; k++ )
        {
            j = cornerpos[k];
            int score = prev[j];
            if( !nonmax_suppression ||
               (score > prev[j+1] && score > prev[j-1] &&
                score > pprev[j-1] && score > pprev[j] && score > pprev[j+1] &&
                score > curr[j-1] && score > curr[j] && score > curr[j+1]) )
            {
                keypoints.push_back(KeyPoint((float)j, (float)(i-1), 7.f, -1, (float)score));
            }
        }
    }
}

void FASTX2(InputArray _img, std::vector<KeyPoint>& keypoints, int threshold, bool nonmax_suppression, int type)
{
  //switch(type) {
  //  case My_FastFeatureDetector::TYPE_5_8:
  //    FAST_t<8>(_img, keypoints, threshold, nonmax_suppression);
  //    break;
  //  case My_FastFeatureDetector::TYPE_7_12:
  //    FAST_t<12>(_img, keypoints, threshold, nonmax_suppression);
  //    break;
  //  case My_FastFeatureDetector::TYPE_9_16:
      FAST_t<16>(_img, keypoints, threshold, nonmax_suppression);
  //    break;
  //}
}

void FAST2(InputArray _img, std::vector<KeyPoint>& keypoints, int threshold, bool nonmax_suppression)
{
    FASTX2(_img, keypoints, threshold, nonmax_suppression, My_FastFeatureDetector::TYPE_9_16);
}

/*
 *   My_FastFeatureDetector
 */
My_FastFeatureDetector::My_FastFeatureDetector( int _threshold, bool _nonmaxSuppression )
    : threshold(_threshold), nonmaxSuppression(_nonmaxSuppression)
{}

void My_FastFeatureDetector::detectImpl( const Mat& image, vector<KeyPoint>& keypoints, const Mat& mask ) const
{
    //std::cout << "Should be using this!!" <<endl;
    Mat grayImage = image;
    if( image.type() != CV_8U ) cvtColor( image, grayImage, CV_BGR2GRAY );
    FAST2( grayImage, keypoints, threshold, nonmaxSuppression );
    KeyPointsFilter::runByPixelsMask( keypoints, mask );
}



//Copy (x,y) location of descriptor matches found from KeyPoint data structures into Point2f vectors
static void matches2points(const vector<DMatch>& matches, const vector<KeyPoint>& kpts_train,
                    const vector<KeyPoint>& kpts_query, vector<Point2f>& pts_train, vector<Point2f>& pts_query)
{
  pts_train.clear();
  pts_query.clear();
  pts_train.reserve(matches.size());
  pts_query.reserve(matches.size());
  for (size_t i = 0; i < matches.size(); i++)
  {
    const DMatch& match = matches[i];
    pts_query.push_back(kpts_query[match.queryIdx].pt);
    pts_train.push_back(kpts_train[match.trainIdx].pt);
  }

}

static double match(const vector<KeyPoint>& /*kpts_train*/, const vector<KeyPoint>& /*kpts_query*/, DescriptorMatcher& matcher,
            const Mat& train, const Mat& query, vector<DMatch>& matches)
{

  double t = (double)getTickCount();
  matcher.match(query, train, matches); //Using features2d
  return ((double)getTickCount() - t) / getTickFrequency();
}

static void help()
{
       cout << "This program shows how to use BRIEF descriptor to match points in features2d" << endl <<
               "It takes in two images, finds keypoints and matches them displaying matches and final homography warped results" << endl <<
                "Usage: " << endl <<
                    "image1 image2 " << endl <<
                "Example: " << endl <<
                    "box.png box_in_scene.png " << endl;
}

const char* keys =
{
    "{1|  |box.png               |the first image}"
    "{2|  |box_in_scene.png|the second image}"
};

int main(int argc, const char ** argv)
{

  //help();
  CommandLineParser parser(argc, argv, keys);
  string im1_name = parser.get<string>("1");
  string im2_name = parser.get<string>("2");

  Mat im1 = imread(im1_name, CV_LOAD_IMAGE_GRAYSCALE);
  Mat im2 = imread(im2_name, CV_LOAD_IMAGE_GRAYSCALE);

  if (im1.empty() || im2.empty())
  {
    cout << "could not open one of the images..." << endl;
    cout << "the cmd parameters have next current value: " << endl;
    parser.printParams();
    return 1;
  }

  //double t = (double)getTickCount();

  My_FastFeatureDetector detector(50);
  BriefDescriptorExtractor extractor(32); //this is really 32 x 8 matches since they are binary matches packed into bytes

  vector<KeyPoint> kpts_1, kpts_2;
  detector.detectImpl(im1, kpts_1);
  detector.detectImpl(im2, kpts_2);

  //t = ((double)getTickCount() - t) / getTickFrequency();

  cout << "found " << kpts_1.size() << " keypoints in " << im1_name << endl << "fount " << kpts_2.size()
      << " keypoints in " << im2_name << endl;

  Mat desc_1, desc_2;

  //cout << "computing descriptors..." << endl;

  //t = (double)getTickCount();

  extractor.compute(im1, kpts_1, desc_1);
  extractor.compute(im2, kpts_2, desc_2);

  //t = ((double)getTickCount() - t) / getTickFrequency();

  //cout << "done computing descriptors... took " << t << " seconds" << endl;

  //Do matching using features2d
  cout << "matching with BruteForceMatcher<Hamming>" << endl;
  BFMatcher matcher_popcount(NORM_HAMMING);
  vector<DMatch> matches_popcount;
  double pop_time = match(kpts_1, kpts_2, matcher_popcount, desc_1, desc_2, matches_popcount);
  cout << "done BruteForceMatcher<Hamming> matching. took " << pop_time << " seconds" << endl;

  //vector<Point2f> mpts_1, mpts_2;
  //matches2points(matches_popcount, kpts_1, kpts_2, mpts_1, mpts_2); //Extract a list of the (x,y) location of the matches
  //cout << "Here 1"<< endl;
  //vector<char> outlier_mask;
  //cout << "Here 2"<< endl;
  //Mat H = findHomography(mpts_2, mpts_1, RANSAC, 1, outlier_mask);

  //Mat outimg;
  //cout << "Here 3"<< endl;
  //drawMatches(im2, kpts_2, im1, kpts_1, matches_popcount, outimg, Scalar::all(-1), Scalar::all(-1), outlier_mask);
  //imshow("matches - popcount - outliers removed", outimg);

  //Mat warped;
  //Mat diff;
  //warpPerspective(im2, warped, H, im1.size());
  //imshow("warped", warped);
  //absdiff(im1,warped,diff);
  //imshow("diff", diff);
  //waitKey();
  return 0;
}

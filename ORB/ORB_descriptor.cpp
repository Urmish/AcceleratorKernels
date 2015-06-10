/**
 * @file SURF_descriptor
 * @brief SURF detector + descritpor + BruteForce Matcher + drawing matches with OpenCV functions
 * @author A. Huaman
 */

#include "precomp.hpp"
#include <stdio.h>
#include <iostream>
#include "opencv2/core/core.hpp"
//#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/features2d.hpp"

using namespace cv;
const float HARRIS_K = 0.04f;
const int DESCRIPTOR_SIZE = 32;

#define My_Round(x) (((x) >= (int)(x)+0.5)? (int)(x)+1:(int)(x))

#define CV_ELEM_SIZE(type) \
     (CV_MAT_CN(type) << ((((sizeof(size_t)/4+1)*16384|0x3a50) >> CV_MAT_DEPTH(type)*2) & 3))

void *memcpy2(void *dst, const void *src, size_t len)
{
        size_t i;

        /*
         * memcpy does not support overlapping buffers, so always do it
         * forwards. (Don't change this without adjusting memmove.)
         *
         * For speedy copying, optimize the common case where both pointers
         * and the length are word-aligned, and copy word-at-a-time instead
         * of byte-at-a-time. Otherwise, copy by bytes.
         *
         * The alignment logic below should be portable. We rely on
         * the compiler to be reasonably intelligent about optimizing
         * the divides and modulos out. Fortunately, it is.
         */

        if ((uintptr_t)dst % sizeof(long) == 0 &&
            (uintptr_t)src % sizeof(long) == 0 &&
            len % sizeof(long) == 0) {

                long *d = (long *)dst;
                const long *s = (long *)src;

                for (i=0; i<len/sizeof(long); i++) {
                        d[i] = s[i];
                }
        }
        else {
                char *d = (char *)dst;
                const char *s = (char *)src;

                for (i=0; i<len; i++) {
                        d[i] = s[i];
                }
        }

        return dst;
}


class CV_EXPORTS FilterEngine2
{
public:
    //! the default constructor
    FilterEngine2();
    //! the full constructor. Either _filter2D or both _rowFilter and _columnFilter must be non-empty.
    FilterEngine2(const Ptr<BaseFilter>& _filter2D,
                 const Ptr<BaseRowFilter>& _rowFilter,
                 const Ptr<BaseColumnFilter>& _columnFilter,
                 int srcType, int dstType, int bufType,
                 int _rowBorderType=BORDER_REPLICATE,
                 int _columnBorderType=-1,
                 const Scalar& _borderValue=Scalar());
    //! the destructor
    virtual ~FilterEngine2();
    //! reinitializes the engine. The previously assigned filters are released.
    void init(const Ptr<BaseFilter>& _filter2D,
              const Ptr<BaseRowFilter>& _rowFilter,
              const Ptr<BaseColumnFilter>& _columnFilter,
              int srcType, int dstType, int bufType,
              int _rowBorderType=BORDER_REPLICATE, int _columnBorderType=-1,
              const Scalar& _borderValue=Scalar());
    //! starts filtering of the specified ROI of an image of size wholeSize.
    virtual int start(Size wholeSize, Rect roi, int maxBufRows=-1);
    //! starts filtering of the specified ROI of the specified image.
    virtual int start(const Mat& src, const Rect& srcRoi=Rect(0,0,-1,-1),
                      bool isolated=false, int maxBufRows=-1);
    //! processes the next srcCount rows of the image.
    virtual int proceed(const uchar* src, int srcStep, int srcCount,
                        uchar* dst, int dstStep);
    //! applies filter to the specified ROI of the image. if srcRoi=(0,0,-1,-1), the whole image is filtered.
    virtual void apply( const Mat& src, Mat& dst,
                        const Rect& srcRoi=Rect(0,0,-1,-1),
                        Point dstOfs=Point(0,0),
                        bool isolated=false);
    //! returns true if the filter is separable
    bool isSeparable() const { return (const BaseFilter*)filter2D == 0; }
    //bool isSeparable() const { return 0; }
    //! returns the number
    int remainingInputRows() const;
    int remainingOutputRows() const;

    int srcType, dstType, bufType;
    Size ksize;
    Point anchor;
    int maxWidth;
    Size wholeSize;
    Rect roi;
    int dx1, dx2;
    int rowBorderType, columnBorderType;
    vector<int> borderTab;
    int borderElemSize;
    vector<uchar> ringBuf;
    vector<uchar> srcRow;
    vector<uchar> constBorderValue;
    vector<uchar> constBorderRow;
    int bufStep, startY, startY0, endY, rowCount, dstY;
    vector<uchar*> rows;

    Ptr<BaseFilter> filter2D;
    Ptr<BaseRowFilter> rowFilter;
    Ptr<BaseColumnFilter> columnFilter;
};



FilterEngine2::FilterEngine2()
{
    srcType = dstType = bufType = -1;
    rowBorderType = columnBorderType = BORDER_REPLICATE;
    bufStep = startY = startY0 = endY = rowCount = dstY = 0;
    maxWidth = 0;

    wholeSize = Size(-1,-1);
}


FilterEngine2::FilterEngine2( const Ptr<BaseFilter>& _filter2D,
                            const Ptr<BaseRowFilter>& _rowFilter,
                            const Ptr<BaseColumnFilter>& _columnFilter,
                            int _srcType, int _dstType, int _bufType,
                            int _rowBorderType, int _columnBorderType,
                            const Scalar& _borderValue )
{
    init(_filter2D, _rowFilter, _columnFilter, _srcType, _dstType, _bufType,
         _rowBorderType, _columnBorderType, _borderValue);
}

FilterEngine2::~FilterEngine2()
{
}


void FilterEngine2::init( const Ptr<BaseFilter>& _filter2D,
                         const Ptr<BaseRowFilter>& _rowFilter,
                         const Ptr<BaseColumnFilter>& _columnFilter,
                         int _srcType, int _dstType, int _bufType,
                         int _rowBorderType, int _columnBorderType,
                         const Scalar& _borderValue )
{
    _srcType = CV_MAT_TYPE(_srcType);
    _bufType = CV_MAT_TYPE(_bufType);
    _dstType = CV_MAT_TYPE(_dstType);

    srcType = _srcType;
    int srcElemSize = (int)getElemSize(srcType);
    dstType = _dstType;
    bufType = _bufType;

    filter2D = _filter2D;
    rowFilter = _rowFilter;
    columnFilter = _columnFilter;

    if( _columnBorderType < 0 )
        _columnBorderType = _rowBorderType;

    rowBorderType = _rowBorderType;
    columnBorderType = _columnBorderType;

    CV_Assert( columnBorderType != BORDER_WRAP );

    if( isSeparable() )
    {
        CV_Assert( !rowFilter.empty() && !columnFilter.empty() );
        ksize = Size(rowFilter->ksize, columnFilter->ksize);
        anchor = Point(rowFilter->anchor, columnFilter->anchor);
    }
    else
    {
        CV_Assert( bufType == srcType );
        ksize = filter2D->ksize;
        anchor = filter2D->anchor;
    }

    CV_Assert( 0 <= anchor.x && anchor.x < ksize.width &&
               0 <= anchor.y && anchor.y < ksize.height );

    borderElemSize = srcElemSize/(CV_MAT_DEPTH(srcType) >= CV_32S ? sizeof(int) : 1);
    int borderLength = std::max(ksize.width - 1, 1);
    borderTab.resize(borderLength*borderElemSize);

    maxWidth = bufStep = 0;
    constBorderRow.clear();

    if( rowBorderType == BORDER_CONSTANT || columnBorderType == BORDER_CONSTANT )
    {
        constBorderValue.resize(srcElemSize*borderLength);
        int srcType1 = CV_MAKETYPE(CV_MAT_DEPTH(srcType), MIN(CV_MAT_CN(srcType), 4));
        scalarToRawData(_borderValue, &constBorderValue[0], srcType1,
                        borderLength*CV_MAT_CN(srcType));
    }

    wholeSize = Size(-1,-1);
}

static const int VEC_ALIGN = CV_MALLOC_ALIGN;

int FilterEngine2::start(Size _wholeSize, Rect _roi, int _maxBufRows)
{
    int i, j;

    wholeSize = _wholeSize;
    roi = _roi;
    CV_Assert( roi.x >= 0 && roi.y >= 0 && roi.width >= 0 && roi.height >= 0 &&
        roi.x + roi.width <= wholeSize.width &&
        roi.y + roi.height <= wholeSize.height );

    int esz = (int)getElemSize(srcType);
    int bufElemSize = (int)getElemSize(bufType);
    const uchar* constVal = !constBorderValue.empty() ? &constBorderValue[0] : 0;

    if( _maxBufRows < 0 )
        _maxBufRows = ksize.height + 3;
    _maxBufRows = std::max(_maxBufRows, std::max(anchor.y, ksize.height-anchor.y-1)*2+1);

    if( maxWidth < roi.width || _maxBufRows != (int)rows.size() )
    {
        rows.resize(_maxBufRows);
        maxWidth = std::max(maxWidth, roi.width);
        int cn = CV_MAT_CN(srcType);
        srcRow.resize(esz*(maxWidth + ksize.width - 1));
        if( columnBorderType == BORDER_CONSTANT )
        {
            constBorderRow.resize(getElemSize(bufType)*(maxWidth + ksize.width - 1 + VEC_ALIGN));
            uchar *dst = alignPtr(&constBorderRow[0], VEC_ALIGN);
	    uchar *tdst;
            int n = (int)constBorderValue.size(), N;
            N = (maxWidth + ksize.width - 1)*esz;
            tdst = isSeparable() ? &srcRow[0] : dst;

            for( i = 0; i < N; i += n )
            {
                n = std::min( n, N - i );
                for(j = 0; j < n; j++)
                    tdst[i+j] = constVal[j];
            }

            if( isSeparable() )
                (*rowFilter)(&srcRow[0], dst, maxWidth, cn);
        }

        int maxBufStep = bufElemSize*(int)alignSize(maxWidth +
            (!isSeparable() ? ksize.width - 1 : 0),VEC_ALIGN);
        ringBuf.resize(maxBufStep*rows.size()+VEC_ALIGN);
    }

    // adjust bufstep so that the used part of the ring buffer stays compact in memory
    bufStep = bufElemSize*(int)alignSize(roi.width + (!isSeparable() ? ksize.width - 1 : 0),16);

    dx1 = std::max(anchor.x - roi.x, 0);
    dx2 = std::max(ksize.width - anchor.x - 1 + roi.x + roi.width - wholeSize.width, 0);

    // recompute border tables
    if( dx1 > 0 || dx2 > 0 )
    {
        if( rowBorderType == BORDER_CONSTANT )
        {
            int nr = isSeparable() ? 1 : (int)rows.size();
            for( i = 0; i < nr; i++ )
            {
                uchar* dst = isSeparable() ? &srcRow[0] : alignPtr(&ringBuf[0],VEC_ALIGN) + bufStep*i;
                memcpy( dst, constVal, dx1*esz );
                memcpy( dst + (roi.width + ksize.width - 1 - dx2)*esz, constVal, dx2*esz );
            }
        }
        else
        {
            int xofs1 = std::min(roi.x, anchor.x) - roi.x;

            int btab_esz = borderElemSize, wholeWidth = wholeSize.width;
            int* btab = (int*)&borderTab[0];

            for( i = 0; i < dx1; i++ )
            {
                int p0 = (borderInterpolate(i-dx1, wholeWidth, rowBorderType) + xofs1)*btab_esz;
                for( j = 0; j < btab_esz; j++ )
                    btab[i*btab_esz + j] = p0 + j;
            }

            for( i = 0; i < dx2; i++ )
            {
                int p0 = (borderInterpolate(wholeWidth + i, wholeWidth, rowBorderType) + xofs1)*btab_esz;
                for( j = 0; j < btab_esz; j++ )
                    btab[(i + dx1)*btab_esz + j] = p0 + j;
            }
        }
    }

    rowCount = dstY = 0;
    startY = startY0 = std::max(roi.y - anchor.y, 0);
    endY = std::min(roi.y + roi.height + ksize.height - anchor.y - 1, wholeSize.height);
    if( !columnFilter.empty() )
        columnFilter->reset();
    if( !filter2D.empty() )
        filter2D->reset();

    return startY;
}


int FilterEngine2::start(const Mat& src, const Rect& _srcRoi,
                        bool isolated, int maxBufRows)
{
    Rect srcRoi = _srcRoi;

    if( srcRoi == Rect(0,0,-1,-1) )
        srcRoi = Rect(0,0,src.cols,src.rows);

    CV_Assert( srcRoi.x >= 0 && srcRoi.y >= 0 &&
        srcRoi.width >= 0 && srcRoi.height >= 0 &&
        srcRoi.x + srcRoi.width <= src.cols &&
        srcRoi.y + srcRoi.height <= src.rows );

    Point ofs;
    Size wsz(src.cols, src.rows);
    if( !isolated )
        src.locateROI( wsz, ofs );
    start( wsz, srcRoi + ofs, maxBufRows );

    return startY - ofs.y;
}


int FilterEngine2::remainingInputRows() const
{
    return endY - startY - rowCount;
}

int FilterEngine2::remainingOutputRows() const
{
    return roi.height - dstY;
}

int FilterEngine2::proceed( const uchar* src, int srcstep, int count,
                           uchar* dst, int dststep )
{
    //CV_Assert( wholeSize.width > 0 && wholeSize.height > 0 );

    const int *btab = &borderTab[0];
    int esz = (int)getElemSize(srcType); 
    int btab_esz = borderElemSize;
    uchar** brows = &rows[0];
    int bufRows = (int)rows.size(); 
    int cn = CV_MAT_CN(bufType);
    int width = roi.width, kwidth = ksize.width; 
    int kheight = ksize.height, ay = anchor.y;
    int _dx1 = dx1, _dx2 = dx2;
    int width1 = roi.width + kwidth - 1; 
    int xofs1 = std::min(roi.x, anchor.x); 
    bool isSep = isSeparable();
  
    bool makeBorder = (_dx1 > 0 || _dx2 > 0) && rowBorderType != BORDER_CONSTANT;
    int dy = 0, i = 0;

    src -= xofs1*esz;
    count = std::min(count, remainingInputRows());

    CV_Assert( src && dst && count > 0 );

    //printf("dststep - %d\n",dststep);
    for(;; dst += dststep*i, dy += i)
    {
        int dcount = bufRows - ay - startY - rowCount + roi.y;
        dcount = dcount > 0 ? dcount : bufRows - kheight + 1;
        dcount = std::min(dcount, count);
        count -= dcount;
        for( ; dcount-- > 0; src += srcstep )
        {
            int bi = (startY - startY0 + rowCount) % bufRows;
            uchar* brow = alignPtr(&ringBuf[0], VEC_ALIGN) + bi*bufStep;
            uchar* row = isSep ? &srcRow[0] : brow;

            if( ++rowCount > bufRows )
            {
                --rowCount;
                ++startY;
            }

            memcpy( row + _dx1*esz, src, (width1 - _dx2 - _dx1)*esz );

           if( makeBorder )
            {
                if( btab_esz*(int)sizeof(int) == esz ) //Sizof(int) Input to Kernel
                {
                    const int* isrc = (const int*)src;
                    int* irow = (int*)row;

                    for( i = 0; i < _dx1*btab_esz; i++ )
                        irow[i] = isrc[btab[i]];
                    for( i = 0; i < _dx2*btab_esz; i++ )
                        irow[i + (width1 - _dx2)*btab_esz] = isrc[btab[i+_dx1*btab_esz]];
                }
                else
                {
                    for( i = 0; i < _dx1*esz; i++ )
                        row[i] = src[btab[i]];
                    for( i = 0; i < _dx2*esz; i++ )
                        row[i + (width1 - _dx2)*esz] = src[btab[i+_dx1*esz]];
                }
            }

            //if( isSep )
            //    (*rowFilter)(row, brow, width, CV_MAT_CN(srcType));
        }
	
        int max_i = std::min(bufRows, roi.height - (dstY + dy) + (kheight - 1));
        for( i = 0; i < max_i; i++ )
        {
            int srcY = borderInterpolate(dstY + dy + i + roi.y - ay,
                            wholeSize.height, columnBorderType);


	    int srcY = p;

            if( srcY < 0 ) // can happen only with constant border type
                brows[i] = alignPtr(&constBorderRow[0], VEC_ALIGN);
            else
            {
                CV_Assert( srcY >= startY );
                if( srcY >= startY + rowCount )
                    break;
                int bi = (srcY - startY0) % bufRows;
                brows[i] = (uchar *) ( ((long int)&ringBuf[0] + VEC_ALIGN - 1) & ~(long int)(VEC_ALIGN-1) ) + bi*bufStep    ;
            }
        }
        if( i < kheight )
            break;
        i -= kheight - 1;
        //if( isSeparable() )
        //    (*columnFilter)((const uchar**)brows, dst, dststep, i, roi.width*cn);
        //else
            (*filter2D)((const uchar**)brows, dst, dststep, i, roi.width, cn);
    }
    //printf("i - %d\n",i);

    dstY += dy;
    CV_Assert( dstY <= roi.height );
    return dy;
}


void FilterEngine2::apply(const Mat& src, Mat& dst,
    const Rect& _srcRoi, Point dstOfs, bool isolated)
{
    CV_Assert( src.type() == srcType && dst.type() == dstType );

    //printf("APPLY 1\n");
    Rect srcRoi = _srcRoi;
    if( srcRoi == Rect(0,0,-1,-1) )
        srcRoi = Rect(0,0,src.cols,src.rows);

    //printf("APPLY 2\n");
    if( srcRoi.area() == 0 )
        return;

    CV_Assert( dstOfs.x >= 0 && dstOfs.y >= 0 &&
        dstOfs.x + srcRoi.width <= dst.cols &&
        dstOfs.y + srcRoi.height <= dst.rows );

    //printf("APPLY 3\n");
    int y = start(src, srcRoi, isolated);
    //printf("Calling Proceed\n");
    proceed( src.data + y*src.step, (int)src.step, endY - startY,
             dst.data + dstOfs.y*dst.step + dstOfs.x*dst.elemSize(), (int)dst.step );
}


CV_EXPORTS Ptr<FilterEngine2> createLinearFilter2(int srcType, int dstType,
                 InputArray kernel, Point _anchor=Point(-1,-1),
                 double delta=0, int rowBorderType=BORDER_DEFAULT,
                 int columnBorderType=-1, const Scalar& borderValue=Scalar());





Ptr<FilterEngine2> createLinearFilter2( int _srcType, int _dstType,
                                              InputArray filter_kernel,
                                              Point _anchor, double _delta,
                                              int _rowBorderType, int _columnBorderType,
                                              const Scalar& _borderValue )
{
    Mat _kernel = filter_kernel.getMat();
    _srcType = CV_MAT_TYPE(_srcType);
    _dstType = CV_MAT_TYPE(_dstType);
    int cn = CV_MAT_CN(_srcType);
    CV_Assert( cn == CV_MAT_CN(_dstType) );

    Mat kernel = _kernel;
    int bits = 0;

    /*int sdepth = CV_MAT_DEPTH(_srcType), ddepth = CV_MAT_DEPTH(_dstType);
    int ktype = _kernel.depth() == CV_32S ? KERNEL_INTEGER : getKernelType(_kernel, _anchor);
    if( sdepth == CV_8U && (ddepth == CV_8U || ddepth == CV_16S) &&
        _kernel.rows*_kernel.cols <= (1 << 10) )
    {
        bits = (ktype & KERNEL_INTEGER) ? 0 : 11;
        _kernel.convertTo(kernel, CV_32S, 1 << bits);
    }*/

    Ptr<BaseFilter> _filter2D = getLinearFilter(_srcType, _dstType,
        kernel, _anchor, _delta, bits);

    return Ptr<FilterEngine2>(new FilterEngine2(_filter2D, Ptr<BaseRowFilter>(0),
        Ptr<BaseColumnFilter>(0), _srcType, _dstType, _srcType,
        _rowBorderType, _columnBorderType, _borderValue ));
}



Ptr<FilterEngine2> createGaussianFilter2(int type, Size ksize, double sigma1, double sigma2, int borderType    )
{
    int depth = CV_MAT_DEPTH(type);
    if( sigma2 <= 0 )
        sigma2 = sigma1;

    // automatic detection of kernel size from sigma
    if( ksize.width <= 0 && sigma1 > 0 )
        ksize.width = cvRound(sigma1*(depth == CV_8U ? 3 : 4)*2 + 1)|1;
    if( ksize.height <= 0 && sigma2 > 0 )
        ksize.height = cvRound(sigma2*(depth == CV_8U ? 3 : 4)*2 + 1)|1;

    CV_Assert( ksize.width > 0 && ksize.width % 2 == 1 &&
        ksize.height > 0 && ksize.height % 2 == 1 );

    sigma1 = std::max( sigma1, 0. );
    sigma2 = std::max( sigma2, 0. );

    Mat kx = getGaussianKernel( ksize.width, sigma1, std::max(depth, CV_32F) );
    //Mat ky;
    //if( ksize.height == ksize.width && std::abs(sigma1 - sigma2) < DBL_EPSILON )
    //    ky = kx;
    //else
    //    ky = getGaussianKernel( ksize.height, sigma2, std::max(depth, CV_32F) );

    return createLinearFilter2( type, type, kx, Point(-1,-1), 0, borderType );
}


void GaussianBlur2( InputArray _src, OutputArray _dst, Size ksize,
                   double sigma1, double sigma2,
                   int borderType )
{
    Mat src = _src.getMat();
    _dst.create( src.size(), src.type() );
    Mat dst = _dst.getMat();

    if( borderType != BORDER_CONSTANT )
    {
        if( src.rows == 1 )
            ksize.height = 1;
        if( src.cols == 1 )
            ksize.width = 1;
    }

    if( ksize.width == 1 && ksize.height == 1 )
    {
        src.copyTo(dst);
        return;
    }


    Ptr<FilterEngine2> f = createGaussianFilter2( src.type(), ksize, sigma1, sigma2, borderType );
    f->apply( src, dst );
};

class ORB2 : public Feature2D
//class ORB2 : public Feature2D
{
public:
    // the size of the signature in bytes
    enum { kBytes = 32, HARRIS_SCORE=0, FAST_SCORE=1 };

    CV_WRAP explicit ORB2(int nfeatures = 500, float scaleFactor = 1.2f, int nlevels = 8, int edgeThreshold = 31,
        int firstLevel = 0, int WTA_K=2, int scoreType=ORB2::HARRIS_SCORE, int patchSize=31 );

    // returns the descriptor size in bytes
    int descriptorSize() const;
    // returns the descriptor type
    int descriptorType() const;

    // Compute the ORB2 features and descriptors on an image
    void operator()(InputArray image, InputArray mask, vector<KeyPoint>& keypoints) const;

    // Compute the ORB2 features and descriptors on an image
    void my_operator( InputArray image, InputArray mask, vector<KeyPoint>& keypoints,
                     OutputArray descriptors, bool useProvidedKeypoints=false ) const;
    void operator()( InputArray image, InputArray mask, vector<KeyPoint>& keypoints,
                     OutputArray descriptors, bool useProvidedKeypoints=false ) const;

    void computeImpl( const Mat& image, vector<KeyPoint>& keypoints, Mat& descriptors ) const;
    void detectImpl( const Mat& image, vector<KeyPoint>& keypoints, const Mat& mask=Mat() ) const;

    CV_PROP_RW int nfeatures;
    CV_PROP_RW double scaleFactor;
    CV_PROP_RW int nlevels;
    CV_PROP_RW int edgeThreshold;
    CV_PROP_RW int firstLevel;
    CV_PROP_RW int WTA_K;
    CV_PROP_RW int scoreType;
    CV_PROP_RW int patchSize;
};
/**
 * Function that computes the Harris responses in a
 * blockSize x blockSize patch at given points in an image
 */
static void
HarrisResponses(const Mat& img, vector<KeyPoint>& pts, int blockSize, float harris_k)
{
    CV_Assert( img.type() == CV_8UC1 && blockSize*blockSize <= 2048 );

    size_t ptidx, ptsize = pts.size();

    const uchar* ptr00 = img.ptr<uchar>(); //Replace this also
    int step = (int)(img.step/img.elemSize1()); //Need to replace this function call
    int r = blockSize/2;

    float scale = (1 << 2) * blockSize * 255.0f;
    scale = 1.0f / scale;
    float scale_sq_sq = scale * scale * scale * scale;

    AutoBuffer<int> ofsbuf(blockSize*blockSize); //We can make this allocation outside and pass a pointer. This allows us to optimize this entire thing as an accelerator
    int* ofs = ofsbuf;
    for( int i = 0; i < blockSize; i++ )
        for( int j = 0; j < blockSize; j++ )
            ofs[i*blockSize + j] = (int)(i*step + j);

    for( ptidx = 0; ptidx < ptsize; ptidx++ )
    {
        int x0 = cvRound(pts[ptidx].pt.x - r);//Need a Macro For This
        int y0 = cvRound(pts[ptidx].pt.y - r);//Need a Macro For This

        const uchar* ptr0 = ptr00 + y0*step + x0; //Need To Change This
        int a = 0, b = 0, c = 0;

        for( int k = 0; k < blockSize*blockSize; k++ )
        {
            const uchar* ptr = ptr0 + ofs[k];
            int Ix = (ptr[1] - ptr[-1])*2 + (ptr[-step+1] - ptr[-step-1]) + (ptr[step+1] - ptr[step-1]);
            int Iy = (ptr[step] - ptr[-step])*2 + (ptr[step-1] - ptr[-step-1]) + (ptr[step+1] - ptr[-step+1]);
            a += Ix*Ix;
            b += Iy*Iy;
            c += Ix*Iy;
        }
        pts[ptidx].response = ((float)a * b - (float)c * c -
                               harris_k * ((float)a + b) * ((float)a + b))*scale_sq_sq;
    }
}

///////////////////////////////////////////////////////////////////////////////
static void
//HarrisResponses_kernel(const Mat& img, vector<KeyPoint>& pts, int blockSize, float harris_k)
HarrisResponses_kernel(const uchar* ptr00,int step, int* ofs, int* x_ptr, int* y_ptr, float* response_ptr, int blockSize, float harris_k, int ptsize)
//pts_x, pts_y, pts_responses
{

    size_t ptidx; //, ptsize = pts.size(); //Need to replace this

    //const uchar* ptr00 = img.ptr<uchar>(); //Replace this also
    //int step = (int)(img.step/img.elemSize1()); //Need to replace this function call, pass the value of step
    int r = blockSize/2;

    float scale = (1 << 2) * blockSize * 255.0f;
    scale = 1.0f / scale;
    float scale_sq_sq = scale * scale * scale * scale;

    //AutoBuffer<int> ofsbuf(blockSize*blockSize); //We can make this allocation outside and pass a pointer. This allows us to optimize this entire thing as an accelerator
    //int* ofs = ofsbuf;
    for( int i = 0; i < blockSize; i++ )
        for( int j = 0; j < blockSize; j++ )
            ofs[i*blockSize + j] = (int)(i*step + j);

    for( ptidx = 0; ptidx < ptsize; ptidx++ )
    {
        //int x0 = My_Round(pts[ptidx].pt.x - r);//Need a Macro For This
        int x0 = My_Round(x_ptr[ptidx] - r);//Need a Macro For This
        //int y0 = My_Round(pts[ptidx].pt.y - r);//Need a Macro For This
        int y0 = My_Round(y_ptr[ptidx] - r);//Need a Macro For This

        const uchar* ptr0 = ptr00 + y0*step + x0; //Need To Change This
        int a = 0, b = 0, c = 0;

        for( int k = 0; k < blockSize*blockSize; k++ )
        {
            const uchar* ptr = ptr0 + ofs[k];
            int Ix = (ptr[1] - ptr[-1])*2 + (ptr[-step+1] - ptr[-step-1]) + (ptr[step+1] - ptr[step-1]);
            int Iy = (ptr[step] - ptr[-step])*2 + (ptr[step-1] - ptr[-step-1]) + (ptr[step+1] - ptr[-step+1]);
            a += Ix*Ix;
            b += Iy*Iy;
            c += Ix*Iy;
        }
        //pts[ptidx].response = ((float)a * b - (float)c * c -
        response_ptr[ptidx] = ((float)a * b - (float)c * c -
                               harris_k * ((float)a + b) * ((float)a + b))*scale_sq_sq; //should this be taken out because of type conversion??
    }
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static float IC_Angle(const Mat& image, const int half_k, Point2f pt,
                      const vector<int> & u_max)
{
    int m_01 = 0, m_10 = 0;

    const uchar* center = &image.at<uchar> (cvRound(pt.y), cvRound(pt.x));

    // Treat the center line differently, v=0
    for (int u = -half_k; u <= half_k; ++u)
        m_10 += u * center[u];

    // Go line by line in the circular patch
    int step = (int)image.step1();
    for (int v = 1; v <= half_k; ++v)
    {
        // Proceed over the two lines
        int v_sum = 0;
        int d = u_max[v];
        for (int u = -d; u <= d; ++u)
        {
            int val_plus = center[u + v*step], val_minus = center[u - v*step];
            v_sum += (val_plus - val_minus);
            m_10 += u * (val_plus + val_minus);
        }
        m_01 += v * v_sum;
    }

    return fastAtan2((float)m_01, (float)m_10);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static void computeORB2Descriptor(const KeyPoint& kpt,
                                 const Mat& img, const Point* pattern,
                                 uchar* desc, int dsize, int WTA_K)
{
    float angle = kpt.angle;
    //angle = cvFloor(angle/12)*12.f;
    angle *= (float)(CV_PI/180.f); //Need to precompute a,b and store it in an array
    float a = (float)cos(angle), b = (float)sin(angle);

    const uchar* center = &img.at<uchar>(cvRound(kpt.pt.y), cvRound(kpt.pt.x));
    int step = (int)img.step;//Pass this as a parameter

    float x, y;
    int ix, iy;
#if 1
    #define GET_VALUE(idx) \
           (x = pattern[idx].x*a - pattern[idx].y*b, \
            y = pattern[idx].x*b + pattern[idx].y*a, \
            ix = cvRound(x), \
            iy = cvRound(y), \
            *(center + iy*step + ix) )
#else
    #define GET_VALUE(idx) \
        (x = pattern[idx].x*a - pattern[idx].y*b, \
        y = pattern[idx].x*b + pattern[idx].y*a, \
        ix = cvFloor(x), iy = cvFloor(y), \
        x -= ix, y -= iy, \
        cvRound(center[iy*step + ix]*(1-x)*(1-y) + center[(iy+1)*step + ix]*(1-x)*y + \
                center[iy*step + ix+1]*x*(1-y) + center[(iy+1)*step + ix+1]*x*y))
#endif

    //ACCPOT This comparator logic could be accelerated in hardware
    if( WTA_K == 2 )
    {
        for (int i = 0; i < dsize; ++i, pattern += 16)
        {
            int t0, t1, val;
            t0 = GET_VALUE(0); t1 = GET_VALUE(1);
            val = t0 < t1;
            t0 = GET_VALUE(2); t1 = GET_VALUE(3);
            val |= (t0 < t1) << 1;
            t0 = GET_VALUE(4); t1 = GET_VALUE(5);
            val |= (t0 < t1) << 2;
            t0 = GET_VALUE(6); t1 = GET_VALUE(7);
            val |= (t0 < t1) << 3;
            t0 = GET_VALUE(8); t1 = GET_VALUE(9);
            val |= (t0 < t1) << 4;
            t0 = GET_VALUE(10); t1 = GET_VALUE(11);
            val |= (t0 < t1) << 5;
            t0 = GET_VALUE(12); t1 = GET_VALUE(13);
            val |= (t0 < t1) << 6;
            t0 = GET_VALUE(14); t1 = GET_VALUE(15);
            val |= (t0 < t1) << 7;

            desc[i] = (uchar)val;
        }
    }
    else if( WTA_K == 3 )
    {
        for (int i = 0; i < dsize; ++i, pattern += 12)
        {
            int t0, t1, t2, val;
            t0 = GET_VALUE(0); t1 = GET_VALUE(1); t2 = GET_VALUE(2);
            val = t2 > t1 ? (t2 > t0 ? 2 : 0) : (t1 > t0);

            t0 = GET_VALUE(3); t1 = GET_VALUE(4); t2 = GET_VALUE(5);
            val |= (t2 > t1 ? (t2 > t0 ? 2 : 0) : (t1 > t0)) << 2;

            t0 = GET_VALUE(6); t1 = GET_VALUE(7); t2 = GET_VALUE(8);
            val |= (t2 > t1 ? (t2 > t0 ? 2 : 0) : (t1 > t0)) << 4;

            t0 = GET_VALUE(9); t1 = GET_VALUE(10); t2 = GET_VALUE(11);
            val |= (t2 > t1 ? (t2 > t0 ? 2 : 0) : (t1 > t0)) << 6;

            desc[i] = (uchar)val;
        }
    }
    else if( WTA_K == 4 )
    {
        for (int i = 0; i < dsize; ++i, pattern += 16)
        {
            int t0, t1, t2, t3, u, v, k, val;
            t0 = GET_VALUE(0); t1 = GET_VALUE(1);
            t2 = GET_VALUE(2); t3 = GET_VALUE(3);
            u = 0, v = 2;
            if( t1 > t0 ) t0 = t1, u = 1;
            if( t3 > t2 ) t2 = t3, v = 3;
            k = t0 > t2 ? u : v;
            val = k;

            t0 = GET_VALUE(4); t1 = GET_VALUE(5);
            t2 = GET_VALUE(6); t3 = GET_VALUE(7);
            u = 0, v = 2;
            if( t1 > t0 ) t0 = t1, u = 1;
            if( t3 > t2 ) t2 = t3, v = 3;
            k = t0 > t2 ? u : v;
            val |= k << 2;

            t0 = GET_VALUE(8); t1 = GET_VALUE(9);
            t2 = GET_VALUE(10); t3 = GET_VALUE(11);
            u = 0, v = 2;
            if( t1 > t0 ) t0 = t1, u = 1;
            if( t3 > t2 ) t2 = t3, v = 3;
            k = t0 > t2 ? u : v;
            val |= k << 4;

            t0 = GET_VALUE(12); t1 = GET_VALUE(13);
            t2 = GET_VALUE(14); t3 = GET_VALUE(15);
            u = 0, v = 2;
            if( t1 > t0 ) t0 = t1, u = 1;
            if( t3 > t2 ) t2 = t3, v = 3;
            k = t0 > t2 ? u : v;
            val |= k << 6;

            desc[i] = (uchar)val;
        }
    }
    else
        CV_Error( CV_StsBadSize, "Wrong WTA_K. It can be only 2, 3 or 4." );

    #undef GET_VALUE
}



//static void computeORB2Descriptor_kernel(const KeyPoint& kpt, const Mat& img, const Point* pattern, uchar* desc, int dsize, int WTA_K)
static void computeORB2Descriptor_kernel(float *a, float *b, const uchar* img_data, int step,int step_p0, int *kpt_x, int *kpt_y,  const Point* pattern, uchar* desc_data, int desc_step_p0, int dsize, int WTA_K,int keypoint_size)
{
    //float angle = kpt.angle;
    ////angle = cvFloor(angle/12)*12.f;
    //angle *= (float)(CV_PI/180.f); //Need to precompute a,b and store it in an array
    //float a = (float)cos(angle), b = (float)sin(angle);

    for (int y_k  =0; y_k< keypoint_size; y_k++)
    {
    	//const uchar* center = &img.at<uchar>(My_Round(kpt_y[y_k]), My_Round(kpt_x[y_k]));
	
    	const uchar* center = &(((img_data + step_p0*kpt_y[y_k]))[kpt_x[y_k]]);
	uchar* desc = desc_data + desc_step_p0*y_k;
    //int step = (int)img.step;//Pass this as a parameter

	    float x, y;
	    int ix, iy;
	    //printf("a-%f\n",a[y_k]);
	    //printf("b-%f\n",b[y_k]);
	    //printf("center-%d\n",*center);
#if 1
    #define GET_VALUE(idx) \
           (x = pattern[idx].x*a[y_k] - pattern[idx].y*b[y_k], \
            y = pattern[idx].x*b[y_k] + pattern[idx].y*a[y_k], \
            ix = cvRound(x), \
            iy = cvRound(y), \
            *(center + iy*step + ix) )
#else
    #define GET_VALUE(idx) \
        (x = pattern[idx].x*a - pattern[idx].y*b, \
        y = pattern[idx].x*b + pattern[idx].y*a, \
        ix = cvFloor(x), iy = cvFloor(y), \
        x -= ix, y -= iy, \
        cvRound(center[iy*step + ix]*(1-x)*(1-y) + center[(iy+1)*step + ix]*(1-x)*y + \
                center[iy*step + ix+1]*x*(1-y) + center[(iy+1)*step + ix+1]*x*y))
#endif

    //ACCPOT This comparator logic could be accelerated in hardware
    //printf("%d,%d Start\n",y_k,keypoint_size);
	    for (int i = 0; i < dsize; ++i, pattern += 16)
    	    {
	    	    //printf("p-x %d\n",pattern[0].x);
	    	    //printf("p-y %d\n",pattern[0].y);
    		    //printf("**************%d,%d Start\n",i,dsize);
            	    int t0, t1;
		    uchar  val;
		    //printf("1\n");
        	    t0 = GET_VALUE(0); t1 = GET_VALUE(1);
	            uchar val_0 = t0 < t1;
		    //printf("2\n");
        	    t0 = GET_VALUE(2); t1 = GET_VALUE(3);
	            uchar val_1 = (t0 < t1) << 1;
		    //printf("3\n");
	            t0 = GET_VALUE(4); t1 = GET_VALUE(5);
	            uchar val_2 = (t0 < t1) << 2;
		    //printf("4\n");
	            t0 = GET_VALUE(6); t1 = GET_VALUE(7);
	            uchar val_3 = (t0 < t1) << 3;
	            t0 = GET_VALUE(8); t1 = GET_VALUE(9);
	            uchar val_4 = (t0 < t1) << 4;
	            t0 = GET_VALUE(10); t1 = GET_VALUE(11);
	            uchar val_5 = (t0 < t1) << 5;
	            t0 = GET_VALUE(12); t1 = GET_VALUE(13);
	            uchar val_6 = (t0 < t1) << 6;
	            t0 = GET_VALUE(14); t1 = GET_VALUE(15);
	            uchar val_7 = (t0 < t1) << 7;
		    val = val_0 | val_1 | val_2 | val_3 | val_4 | val_5 | val_6 | val_7;
    		    //printf("**************%d,%d Done\n",i,dsize);
	            desc[i] = val;
    	    }
	pattern = pattern - 16*dsize;
	//printf("%d,%d Done\n",y_k,keypoint_size);
     }
}
///////////////////////////////////////////////////////////////////////////
static void initializeORB2Pattern( const Point* pattern0, vector<Point>& pattern, int ntuples, int tupleSize, int poolSize )
{
    RNG rng(0x12345678);
    int i, k, k1;
    pattern.resize(ntuples*tupleSize);

    for( i = 0; i < ntuples; i++ )
    {
        for( k = 0; k < tupleSize; k++ )
        {
            for(;;)
            {
                int idx = rng.uniform(0, poolSize);
                Point pt = pattern0[idx];
                for( k1 = 0; k1 < k; k1++ )
                    if( pattern[tupleSize*i + k1] == pt )
                        break;
                if( k1 == k )
                {
                    pattern[tupleSize*i + k] = pt;
                    break;
                }
            }
        }
    }
}

static int bit_pattern_31_[256*4] =
{
    8,-3, 9,5/*mean (0), correlation (0)*/,
    4,2, 7,-12/*mean (1.12461e-05), correlation (0.0437584)*/,
    -11,9, -8,2/*mean (3.37382e-05), correlation (0.0617409)*/,
    7,-12, 12,-13/*mean (5.62303e-05), correlation (0.0636977)*/,
    2,-13, 2,12/*mean (0.000134953), correlation (0.085099)*/,
    1,-7, 1,6/*mean (0.000528565), correlation (0.0857175)*/,
    -2,-10, -2,-4/*mean (0.0188821), correlation (0.0985774)*/,
    -13,-13, -11,-8/*mean (0.0363135), correlation (0.0899616)*/,
    -13,-3, -12,-9/*mean (0.121806), correlation (0.099849)*/,
    10,4, 11,9/*mean (0.122065), correlation (0.093285)*/,
    -13,-8, -8,-9/*mean (0.162787), correlation (0.0942748)*/,
    -11,7, -9,12/*mean (0.21561), correlation (0.0974438)*/,
    7,7, 12,6/*mean (0.160583), correlation (0.130064)*/,
    -4,-5, -3,0/*mean (0.228171), correlation (0.132998)*/,
    -13,2, -12,-3/*mean (0.00997526), correlation (0.145926)*/,
    -9,0, -7,5/*mean (0.198234), correlation (0.143636)*/,
    12,-6, 12,-1/*mean (0.0676226), correlation (0.16689)*/,
    -3,6, -2,12/*mean (0.166847), correlation (0.171682)*/,
    -6,-13, -4,-8/*mean (0.101215), correlation (0.179716)*/,
    11,-13, 12,-8/*mean (0.200641), correlation (0.192279)*/,
    4,7, 5,1/*mean (0.205106), correlation (0.186848)*/,
    5,-3, 10,-3/*mean (0.234908), correlation (0.192319)*/,
    3,-7, 6,12/*mean (0.0709964), correlation (0.210872)*/,
    -8,-7, -6,-2/*mean (0.0939834), correlation (0.212589)*/,
    -2,11, -1,-10/*mean (0.127778), correlation (0.20866)*/,
    -13,12, -8,10/*mean (0.14783), correlation (0.206356)*/,
    -7,3, -5,-3/*mean (0.182141), correlation (0.198942)*/,
    -4,2, -3,7/*mean (0.188237), correlation (0.21384)*/,
    -10,-12, -6,11/*mean (0.14865), correlation (0.23571)*/,
    5,-12, 6,-7/*mean (0.222312), correlation (0.23324)*/,
    5,-6, 7,-1/*mean (0.229082), correlation (0.23389)*/,
    1,0, 4,-5/*mean (0.241577), correlation (0.215286)*/,
    9,11, 11,-13/*mean (0.00338507), correlation (0.251373)*/,
    4,7, 4,12/*mean (0.131005), correlation (0.257622)*/,
    2,-1, 4,4/*mean (0.152755), correlation (0.255205)*/,
    -4,-12, -2,7/*mean (0.182771), correlation (0.244867)*/,
    -8,-5, -7,-10/*mean (0.186898), correlation (0.23901)*/,
    4,11, 9,12/*mean (0.226226), correlation (0.258255)*/,
    0,-8, 1,-13/*mean (0.0897886), correlation (0.274827)*/,
    -13,-2, -8,2/*mean (0.148774), correlation (0.28065)*/,
    -3,-2, -2,3/*mean (0.153048), correlation (0.283063)*/,
    -6,9, -4,-9/*mean (0.169523), correlation (0.278248)*/,
    8,12, 10,7/*mean (0.225337), correlation (0.282851)*/,
    0,9, 1,3/*mean (0.226687), correlation (0.278734)*/,
    7,-5, 11,-10/*mean (0.00693882), correlation (0.305161)*/,
    -13,-6, -11,0/*mean (0.0227283), correlation (0.300181)*/,
    10,7, 12,1/*mean (0.125517), correlation (0.31089)*/,
    -6,-3, -6,12/*mean (0.131748), correlation (0.312779)*/,
    10,-9, 12,-4/*mean (0.144827), correlation (0.292797)*/,
    -13,8, -8,-12/*mean (0.149202), correlation (0.308918)*/,
    -13,0, -8,-4/*mean (0.160909), correlation (0.310013)*/,
    3,3, 7,8/*mean (0.177755), correlation (0.309394)*/,
    5,7, 10,-7/*mean (0.212337), correlation (0.310315)*/,
    -1,7, 1,-12/*mean (0.214429), correlation (0.311933)*/,
    3,-10, 5,6/*mean (0.235807), correlation (0.313104)*/,
    2,-4, 3,-10/*mean (0.00494827), correlation (0.344948)*/,
    -13,0, -13,5/*mean (0.0549145), correlation (0.344675)*/,
    -13,-7, -12,12/*mean (0.103385), correlation (0.342715)*/,
    -13,3, -11,8/*mean (0.134222), correlation (0.322922)*/,
    -7,12, -4,7/*mean (0.153284), correlation (0.337061)*/,
    6,-10, 12,8/*mean (0.154881), correlation (0.329257)*/,
    -9,-1, -7,-6/*mean (0.200967), correlation (0.33312)*/,
    -2,-5, 0,12/*mean (0.201518), correlation (0.340635)*/,
    -12,5, -7,5/*mean (0.207805), correlation (0.335631)*/,
    3,-10, 8,-13/*mean (0.224438), correlation (0.34504)*/,
    -7,-7, -4,5/*mean (0.239361), correlation (0.338053)*/,
    -3,-2, -1,-7/*mean (0.240744), correlation (0.344322)*/,
    2,9, 5,-11/*mean (0.242949), correlation (0.34145)*/,
    -11,-13, -5,-13/*mean (0.244028), correlation (0.336861)*/,
    -1,6, 0,-1/*mean (0.247571), correlation (0.343684)*/,
    5,-3, 5,2/*mean (0.000697256), correlation (0.357265)*/,
    -4,-13, -4,12/*mean (0.00213675), correlation (0.373827)*/,
    -9,-6, -9,6/*mean (0.0126856), correlation (0.373938)*/,
    -12,-10, -8,-4/*mean (0.0152497), correlation (0.364237)*/,
    10,2, 12,-3/*mean (0.0299933), correlation (0.345292)*/,
    7,12, 12,12/*mean (0.0307242), correlation (0.366299)*/,
    -7,-13, -6,5/*mean (0.0534975), correlation (0.368357)*/,
    -4,9, -3,4/*mean (0.099865), correlation (0.372276)*/,
    7,-1, 12,2/*mean (0.117083), correlation (0.364529)*/,
    -7,6, -5,1/*mean (0.126125), correlation (0.369606)*/,
    -13,11, -12,5/*mean (0.130364), correlation (0.358502)*/,
    -3,7, -2,-6/*mean (0.131691), correlation (0.375531)*/,
    7,-8, 12,-7/*mean (0.160166), correlation (0.379508)*/,
    -13,-7, -11,-12/*mean (0.167848), correlation (0.353343)*/,
    1,-3, 12,12/*mean (0.183378), correlation (0.371916)*/,
    2,-6, 3,0/*mean (0.228711), correlation (0.371761)*/,
    -4,3, -2,-13/*mean (0.247211), correlation (0.364063)*/,
    -1,-13, 1,9/*mean (0.249325), correlation (0.378139)*/,
    7,1, 8,-6/*mean (0.000652272), correlation (0.411682)*/,
    1,-1, 3,12/*mean (0.00248538), correlation (0.392988)*/,
    9,1, 12,6/*mean (0.0206815), correlation (0.386106)*/,
    -1,-9, -1,3/*mean (0.0364485), correlation (0.410752)*/,
    -13,-13, -10,5/*mean (0.0376068), correlation (0.398374)*/,
    7,7, 10,12/*mean (0.0424202), correlation (0.405663)*/,
    12,-5, 12,9/*mean (0.0942645), correlation (0.410422)*/,
    6,3, 7,11/*mean (0.1074), correlation (0.413224)*/,
    5,-13, 6,10/*mean (0.109256), correlation (0.408646)*/,
    2,-12, 2,3/*mean (0.131691), correlation (0.416076)*/,
    3,8, 4,-6/*mean (0.165081), correlation (0.417569)*/,
    2,6, 12,-13/*mean (0.171874), correlation (0.408471)*/,
    9,-12, 10,3/*mean (0.175146), correlation (0.41296)*/,
    -8,4, -7,9/*mean (0.183682), correlation (0.402956)*/,
    -11,12, -4,-6/*mean (0.184672), correlation (0.416125)*/,
    1,12, 2,-8/*mean (0.191487), correlation (0.386696)*/,
    6,-9, 7,-4/*mean (0.192668), correlation (0.394771)*/,
    2,3, 3,-2/*mean (0.200157), correlation (0.408303)*/,
    6,3, 11,0/*mean (0.204588), correlation (0.411762)*/,
    3,-3, 8,-8/*mean (0.205904), correlation (0.416294)*/,
    7,8, 9,3/*mean (0.213237), correlation (0.409306)*/,
    -11,-5, -6,-4/*mean (0.243444), correlation (0.395069)*/,
    -10,11, -5,10/*mean (0.247672), correlation (0.413392)*/,
    -5,-8, -3,12/*mean (0.24774), correlation (0.411416)*/,
    -10,5, -9,0/*mean (0.00213675), correlation (0.454003)*/,
    8,-1, 12,-6/*mean (0.0293635), correlation (0.455368)*/,
    4,-6, 6,-11/*mean (0.0404971), correlation (0.457393)*/,
    -10,12, -8,7/*mean (0.0481107), correlation (0.448364)*/,
    4,-2, 6,7/*mean (0.050641), correlation (0.455019)*/,
    -2,0, -2,12/*mean (0.0525978), correlation (0.44338)*/,
    -5,-8, -5,2/*mean (0.0629667), correlation (0.457096)*/,
    7,-6, 10,12/*mean (0.0653846), correlation (0.445623)*/,
    -9,-13, -8,-8/*mean (0.0858749), correlation (0.449789)*/,
    -5,-13, -5,-2/*mean (0.122402), correlation (0.450201)*/,
    8,-8, 9,-13/*mean (0.125416), correlation (0.453224)*/,
    -9,-11, -9,0/*mean (0.130128), correlation (0.458724)*/,
    1,-8, 1,-2/*mean (0.132467), correlation (0.440133)*/,
    7,-4, 9,1/*mean (0.132692), correlation (0.454)*/,
    -2,1, -1,-4/*mean (0.135695), correlation (0.455739)*/,
    11,-6, 12,-11/*mean (0.142904), correlation (0.446114)*/,
    -12,-9, -6,4/*mean (0.146165), correlation (0.451473)*/,
    3,7, 7,12/*mean (0.147627), correlation (0.456643)*/,
    5,5, 10,8/*mean (0.152901), correlation (0.455036)*/,
    0,-4, 2,8/*mean (0.167083), correlation (0.459315)*/,
    -9,12, -5,-13/*mean (0.173234), correlation (0.454706)*/,
    0,7, 2,12/*mean (0.18312), correlation (0.433855)*/,
    -1,2, 1,7/*mean (0.185504), correlation (0.443838)*/,
    5,11, 7,-9/*mean (0.185706), correlation (0.451123)*/,
    3,5, 6,-8/*mean (0.188968), correlation (0.455808)*/,
    -13,-4, -8,9/*mean (0.191667), correlation (0.459128)*/,
    -5,9, -3,-3/*mean (0.193196), correlation (0.458364)*/,
    -4,-7, -3,-12/*mean (0.196536), correlation (0.455782)*/,
    6,5, 8,0/*mean (0.1972), correlation (0.450481)*/,
    -7,6, -6,12/*mean (0.199438), correlation (0.458156)*/,
    -13,6, -5,-2/*mean (0.211224), correlation (0.449548)*/,
    1,-10, 3,10/*mean (0.211718), correlation (0.440606)*/,
    4,1, 8,-4/*mean (0.213034), correlation (0.443177)*/,
    -2,-2, 2,-13/*mean (0.234334), correlation (0.455304)*/,
    2,-12, 12,12/*mean (0.235684), correlation (0.443436)*/,
    -2,-13, 0,-6/*mean (0.237674), correlation (0.452525)*/,
    4,1, 9,3/*mean (0.23962), correlation (0.444824)*/,
    -6,-10, -3,-5/*mean (0.248459), correlation (0.439621)*/,
    -3,-13, -1,1/*mean (0.249505), correlation (0.456666)*/,
    7,5, 12,-11/*mean (0.00119208), correlation (0.495466)*/,
    4,-2, 5,-7/*mean (0.00372245), correlation (0.484214)*/,
    -13,9, -9,-5/*mean (0.00741116), correlation (0.499854)*/,
    7,1, 8,6/*mean (0.0208952), correlation (0.499773)*/,
    7,-8, 7,6/*mean (0.0220085), correlation (0.501609)*/,
    -7,-4, -7,1/*mean (0.0233806), correlation (0.496568)*/,
    -8,11, -7,-8/*mean (0.0236505), correlation (0.489719)*/,
    -13,6, -12,-8/*mean (0.0268781), correlation (0.503487)*/,
    2,4, 3,9/*mean (0.0323324), correlation (0.501938)*/,
    10,-5, 12,3/*mean (0.0399235), correlation (0.494029)*/,
    -6,-5, -6,7/*mean (0.0420153), correlation (0.486579)*/,
    8,-3, 9,-8/*mean (0.0548021), correlation (0.484237)*/,
    2,-12, 2,8/*mean (0.0616622), correlation (0.496642)*/,
    -11,-2, -10,3/*mean (0.0627755), correlation (0.498563)*/,
    -12,-13, -7,-9/*mean (0.0829622), correlation (0.495491)*/,
    -11,0, -10,-5/*mean (0.0843342), correlation (0.487146)*/,
    5,-3, 11,8/*mean (0.0929937), correlation (0.502315)*/,
    -2,-13, -1,12/*mean (0.113327), correlation (0.48941)*/,
    -1,-8, 0,9/*mean (0.132119), correlation (0.467268)*/,
    -13,-11, -12,-5/*mean (0.136269), correlation (0.498771)*/,
    -10,-2, -10,11/*mean (0.142173), correlation (0.498714)*/,
    -3,9, -2,-13/*mean (0.144141), correlation (0.491973)*/,
    2,-3, 3,2/*mean (0.14892), correlation (0.500782)*/,
    -9,-13, -4,0/*mean (0.150371), correlation (0.498211)*/,
    -4,6, -3,-10/*mean (0.152159), correlation (0.495547)*/,
    -4,12, -2,-7/*mean (0.156152), correlation (0.496925)*/,
    -6,-11, -4,9/*mean (0.15749), correlation (0.499222)*/,
    6,-3, 6,11/*mean (0.159211), correlation (0.503821)*/,
    -13,11, -5,5/*mean (0.162427), correlation (0.501907)*/,
    11,11, 12,6/*mean (0.16652), correlation (0.497632)*/,
    7,-5, 12,-2/*mean (0.169141), correlation (0.484474)*/,
    -1,12, 0,7/*mean (0.169456), correlation (0.495339)*/,
    -4,-8, -3,-2/*mean (0.171457), correlation (0.487251)*/,
    -7,1, -6,7/*mean (0.175), correlation (0.500024)*/,
    -13,-12, -8,-13/*mean (0.175866), correlation (0.497523)*/,
    -7,-2, -6,-8/*mean (0.178273), correlation (0.501854)*/,
    -8,5, -6,-9/*mean (0.181107), correlation (0.494888)*/,
    -5,-1, -4,5/*mean (0.190227), correlation (0.482557)*/,
    -13,7, -8,10/*mean (0.196739), correlation (0.496503)*/,
    1,5, 5,-13/*mean (0.19973), correlation (0.499759)*/,
    1,0, 10,-13/*mean (0.204465), correlation (0.49873)*/,
    9,12, 10,-1/*mean (0.209334), correlation (0.49063)*/,
    5,-8, 10,-9/*mean (0.211134), correlation (0.503011)*/,
    -1,11, 1,-13/*mean (0.212), correlation (0.499414)*/,
    -9,-3, -6,2/*mean (0.212168), correlation (0.480739)*/,
    -1,-10, 1,12/*mean (0.212731), correlation (0.502523)*/,
    -13,1, -8,-10/*mean (0.21327), correlation (0.489786)*/,
    8,-11, 10,-6/*mean (0.214159), correlation (0.488246)*/,
    2,-13, 3,-6/*mean (0.216993), correlation (0.50287)*/,
    7,-13, 12,-9/*mean (0.223639), correlation (0.470502)*/,
    -10,-10, -5,-7/*mean (0.224089), correlation (0.500852)*/,
    -10,-8, -8,-13/*mean (0.228666), correlation (0.502629)*/,
    4,-6, 8,5/*mean (0.22906), correlation (0.498305)*/,
    3,12, 8,-13/*mean (0.233378), correlation (0.503825)*/,
    -4,2, -3,-3/*mean (0.234323), correlation (0.476692)*/,
    5,-13, 10,-12/*mean (0.236392), correlation (0.475462)*/,
    4,-13, 5,-1/*mean (0.236842), correlation (0.504132)*/,
    -9,9, -4,3/*mean (0.236977), correlation (0.497739)*/,
    0,3, 3,-9/*mean (0.24314), correlation (0.499398)*/,
    -12,1, -6,1/*mean (0.243297), correlation (0.489447)*/,
    3,2, 4,-8/*mean (0.00155196), correlation (0.553496)*/,
    -10,-10, -10,9/*mean (0.00239541), correlation (0.54297)*/,
    8,-13, 12,12/*mean (0.0034413), correlation (0.544361)*/,
    -8,-12, -6,-5/*mean (0.003565), correlation (0.551225)*/,
    2,2, 3,7/*mean (0.00835583), correlation (0.55285)*/,
    10,6, 11,-8/*mean (0.00885065), correlation (0.540913)*/,
    6,8, 8,-12/*mean (0.0101552), correlation (0.551085)*/,
    -7,10, -6,5/*mean (0.0102227), correlation (0.533635)*/,
    -3,-9, -3,9/*mean (0.0110211), correlation (0.543121)*/,
    -1,-13, -1,5/*mean (0.0113473), correlation (0.550173)*/,
    -3,-7, -3,4/*mean (0.0140913), correlation (0.554774)*/,
    -8,-2, -8,3/*mean (0.017049), correlation (0.55461)*/,
    4,2, 12,12/*mean (0.01778), correlation (0.546921)*/,
    2,-5, 3,11/*mean (0.0224022), correlation (0.549667)*/,
    6,-9, 11,-13/*mean (0.029161), correlation (0.546295)*/,
    3,-1, 7,12/*mean (0.0303081), correlation (0.548599)*/,
    11,-1, 12,4/*mean (0.0355151), correlation (0.523943)*/,
    -3,0, -3,6/*mean (0.0417904), correlation (0.543395)*/,
    4,-11, 4,12/*mean (0.0487292), correlation (0.542818)*/,
    2,-4, 2,1/*mean (0.0575124), correlation (0.554888)*/,
    -10,-6, -8,1/*mean (0.0594242), correlation (0.544026)*/,
    -13,7, -11,1/*mean (0.0597391), correlation (0.550524)*/,
    -13,12, -11,-13/*mean (0.0608974), correlation (0.55383)*/,
    6,0, 11,-13/*mean (0.065126), correlation (0.552006)*/,
    0,-1, 1,4/*mean (0.074224), correlation (0.546372)*/,
    -13,3, -9,-2/*mean (0.0808592), correlation (0.554875)*/,
    -9,8, -6,-3/*mean (0.0883378), correlation (0.551178)*/,
    -13,-6, -8,-2/*mean (0.0901035), correlation (0.548446)*/,
    5,-9, 8,10/*mean (0.0949843), correlation (0.554694)*/,
    2,7, 3,-9/*mean (0.0994152), correlation (0.550979)*/,
    -1,-6, -1,-1/*mean (0.10045), correlation (0.552714)*/,
    9,5, 11,-2/*mean (0.100686), correlation (0.552594)*/,
    11,-3, 12,-8/*mean (0.101091), correlation (0.532394)*/,
    3,0, 3,5/*mean (0.101147), correlation (0.525576)*/,
    -1,4, 0,10/*mean (0.105263), correlation (0.531498)*/,
    3,-6, 4,5/*mean (0.110785), correlation (0.540491)*/,
    -13,0, -10,5/*mean (0.112798), correlation (0.536582)*/,
    5,8, 12,11/*mean (0.114181), correlation (0.555793)*/,
    8,9, 9,-6/*mean (0.117431), correlation (0.553763)*/,
    7,-4, 8,-12/*mean (0.118522), correlation (0.553452)*/,
    -10,4, -10,9/*mean (0.12094), correlation (0.554785)*/,
    7,3, 12,4/*mean (0.122582), correlation (0.555825)*/,
    9,-7, 10,-2/*mean (0.124978), correlation (0.549846)*/,
    7,0, 12,-2/*mean (0.127002), correlation (0.537452)*/,
    -1,-6, 0,-11/*mean (0.127148), correlation (0.547401)*/
};


static void makeRandomPattern(int patchSize, Point* pattern, int npoints)
{
    RNG rng(0x34985739); // we always start with a fixed seed,
                         // to make patterns the same on each run
    for( int i = 0; i < npoints; i++ )
    {
        pattern[i].x = rng.uniform(-patchSize/2, patchSize/2+1);
        pattern[i].y = rng.uniform(-patchSize/2, patchSize/2+1);
    }
}


static inline float getScale(int level, int firstLevel, double scaleFactor)
{
    return (float)std::pow(scaleFactor, (double)(level - firstLevel));
}

/** Constructor
 * @param detector_params parameters to use
 */
ORB2::ORB2(int _nfeatures, float _scaleFactor, int _nlevels, int _edgeThreshold,
         int _firstLevel, int _WTA_K, int _scoreType, int _patchSize) :
    nfeatures(_nfeatures), scaleFactor(_scaleFactor), nlevels(_nlevels),
    edgeThreshold(_edgeThreshold), firstLevel(_firstLevel), WTA_K(_WTA_K),
    scoreType(_scoreType), patchSize(_patchSize)
{}


int ORB2::descriptorSize() const
{
    return kBytes;
}

int ORB2::descriptorType() const
{
    return CV_8U;
}

/** Compute the ORB2 features and descriptors on an image
 * @param img the image to compute the features and descriptors on
 * @param mask the mask to apply
 * @param keypoints the resulting keypoints
 */
void ORB2::operator()(InputArray image, InputArray mask, vector<KeyPoint>& keypoints) const
{
    (*this).my_operator(image, mask, keypoints, noArray(), false);
}


/** Compute the ORB2 keypoint orientations
 * @param image the image to compute the features and descriptors on
 * @param integral_image the integral image of the iamge (can be empty, but the computation will be slower)
 * @param scale the scale at which we compute the orientation
 * @param keypoints the resulting keypoints
 */
static void computeOrientation(const Mat& image, vector<KeyPoint>& keypoints,
                               int halfPatchSize, const vector<int>& umax)
{
    // Process each keypoint
    for (vector<KeyPoint>::iterator keypoint = keypoints.begin(),
         keypointEnd = keypoints.end(); keypoint != keypointEnd; ++keypoint)
    {
        keypoint->angle = IC_Angle(image, halfPatchSize, keypoint->pt, umax);
    }
}


/** Compute the ORB2 keypoints on an image
 * @param image_pyramid the image pyramid to compute the features and descriptors on
 * @param mask_pyramid the masks to apply at every level
 * @param keypoints the resulting keypoints, clustered per level
 */
static void computeKeyPoints(const vector<Mat>& imagePyramid,
                             const vector<Mat>& maskPyramid,
                             vector<vector<KeyPoint> >& allKeypoints,
                             int nfeatures, int firstLevel, double scaleFactor,
                             int edgeThreshold, int patchSize, int scoreType )
{
    int nlevels = (int)imagePyramid.size();
    vector<int> nfeaturesPerLevel(nlevels);

    // fill the extractors and descriptors for the corresponding scales
    float factor = (float)(1.0 / scaleFactor);
    float ndesiredFeaturesPerScale = nfeatures*(1 - factor)/(1 - (float)pow((double)factor, (double)nlevels));

    int sumFeatures = 0;
    for( int level = 0; level < nlevels-1; level++ )
    {
        nfeaturesPerLevel[level] = cvRound(ndesiredFeaturesPerScale);
        sumFeatures += nfeaturesPerLevel[level];
        ndesiredFeaturesPerScale *= factor;
    }
    nfeaturesPerLevel[nlevels-1] = std::max(nfeatures - sumFeatures, 0);

    // Make sure we forget about what is too close to the boundary
    //edge_threshold_ = std::max(edge_threshold_, patch_size_/2 + kKernelWidth / 2 + 2);

    // pre-compute the end of a row in a circular patch
    int halfPatchSize = patchSize / 2;
    vector<int> umax(halfPatchSize + 2);

    int v, v0, vmax = cvFloor(halfPatchSize * sqrt(2.f) / 2 + 1);
    int vmin = cvCeil(halfPatchSize * sqrt(2.f) / 2);
    for (v = 0; v <= vmax; ++v)
        umax[v] = cvRound(sqrt((double)halfPatchSize * halfPatchSize - v * v));

    // Make sure we are symmetric
    for (v = halfPatchSize, v0 = 0; v >= vmin; --v)
    {
        while (umax[v0] == umax[v0 + 1])
            ++v0;
        umax[v] = v0;
        ++v0;
    }

    allKeypoints.resize(nlevels);

    for (int level = 0; level < nlevels; ++level)
    {
        int featuresNum = nfeaturesPerLevel[level];
        allKeypoints[level].reserve(featuresNum*2);

        vector<KeyPoint> & keypoints = allKeypoints[level];

        // Detect FAST features, 20 is a good threshold
        FastFeatureDetector fd(20, true);
        fd.detect(imagePyramid[level], keypoints, maskPyramid[level]);

        // Remove keypoints very close to the border
        KeyPointsFilter::runByImageBorder(keypoints, imagePyramid[level].size(), edgeThreshold);

        if( scoreType == ORB2::HARRIS_SCORE )
        {
            // Keep more points than necessary as FAST does not give amazing corners
            KeyPointsFilter::retainBest(keypoints, 2 * featuresNum);

            // Compute the Harris cornerness (better scoring than FAST)
            //HarrisResponses(imagePyramid[level], keypoints, 7, HARRIS_K);
    	    CV_Assert( imagePyramid[level].type() == CV_8UC1 && 7*7 <= 2048 );
	    const uchar* ptr00 = imagePyramid[level].ptr<uchar>();
	    AutoBuffer<int> ofsbuf(7*7); //This could be a potential source of error, what if local stack is not visible?
    	    int step = (int)(imagePyramid[level].step/imagePyramid[level].elemSize1()); //Need to replace this function call, pass the value of step
	    int *x_ptr = (int *)malloc(sizeof(int)*keypoints.size());
	    int *y_ptr = (int *)malloc(sizeof(int)*keypoints.size());
	    float *response_ptr = (float *)malloc(sizeof(float)*keypoints.size());
	    for(int i=0; i<keypoints.size();i++)
	    {
		x_ptr[i] = keypoints[i].pt.x;
		y_ptr[i] = keypoints[i].pt.y;
		response_ptr[i] = keypoints[i].response;
	    }
	    HarrisResponses_kernel(ptr00,step,(int *)ofsbuf, x_ptr, y_ptr, response_ptr,7, HARRIS_K,keypoints.size());
	    for(int i=0; i<keypoints.size();i++)
	    {
		keypoints[i].response = response_ptr[i]; //TODO See if you can imrove upon this. May be pass the sie of keypoint and a keypoint ptr and then increment and sae directly
	    }
	    
        }

        //cull to the final desired level, using the new Harris scores or the original FAST scores.
        KeyPointsFilter::retainBest(keypoints, featuresNum);

        float sf = getScale(level, firstLevel, scaleFactor);

        // Set the level of the coordinates
        for (vector<KeyPoint>::iterator keypoint = keypoints.begin(),
             keypointEnd = keypoints.end(); keypoint != keypointEnd; ++keypoint)
        {
            keypoint->octave = level;
            keypoint->size = patchSize*sf;
        }

        computeOrientation(imagePyramid[level], keypoints, halfPatchSize, umax);
    }
}


/** Compute the ORB2 decriptors
 * @param image the image to compute the features and descriptors on
 * @param integral_image the integral image of the image (can be empty, but the computation will be slower)
 * @param level the scale at which we compute the orientation
 * @param keypoints the keypoints to use
 * @param descriptors the resulting descriptors
 */
static void computeDescriptors(const Mat& image, vector<KeyPoint>& keypoints, Mat& descriptors,
                               const vector<Point>& pattern, int dsize, int WTA_K)
{
    //convert to grayscale if more than one color
    CV_Assert(image.type() == CV_8UC1);
    //create the descriptor mat, keypoints.size() rows, BYTES cols
    descriptors = Mat::zeros((int)keypoints.size(), dsize, CV_8UC1);

    /* Copied from kernel
	float angle = kpt.angle;
    	angle *= (float)(CV_PI/180.f); //Need to precompute a,b and store it in an array
    	float a = (float)cos(angle), b = (float)sin(angle);
    */
    int* kpt_x = (int *)malloc(keypoints.size()*sizeof(int));
    int* kpt_y = (int *)malloc(keypoints.size()*sizeof(int));
    float* kpt_a = (float *)malloc(keypoints.size()*sizeof(float));
    float* kpt_b = (float *)malloc(keypoints.size()*sizeof(float));
    for (size_t i = 0; i < keypoints.size(); i++) //Would need to accelerate this entire loop
    {
	kpt_x[i] = keypoints[i].pt.x;
	kpt_y[i] = keypoints[i].pt.y;	
	float angle = keypoints[i].angle;
    	angle *= (float)(CV_PI/180.f);
    	kpt_a[i] = (float)cos(angle),
	kpt_b[i] = (float)sin(angle);
    }     
    //for (size_t i = 0; i < keypoints.size(); i++) //Would need to accelerate this entire loop
        //computeORB2Descriptor_kernel(keypoints[i], image, &pattern[0], descriptors.ptr((int)i), dsize, WTA_K);
    computeORB2Descriptor_kernel(kpt_a, kpt_b, (uchar* )image.data, image.step, image.step.p[0], kpt_x, kpt_y, &pattern[0], descriptors.data,descriptors.step.p[0], dsize, WTA_K, keypoints.size());
}


/** Compute the ORB2 features and descriptors on an image
 * @param img the image to compute the features and descriptors on
 * @param mask the mask to apply
 * @param keypoints the resulting keypoints
 * @param descriptors the resulting descriptors
 * @param do_keypoints if true, the keypoints are computed, otherwise used as an input
 * @param do_descriptors if true, also computes the descriptors
 */
void ORB2::operator()( InputArray _image, InputArray _mask, vector<KeyPoint>& _keypoints,
                      OutputArray _descriptors, bool useProvidedKeypoints) const
{
}

void ORB2::my_operator( InputArray _image, InputArray _mask, vector<KeyPoint>& _keypoints,
                      OutputArray _descriptors, bool useProvidedKeypoints) const
{
    //std::cout <<"Should be using this!!!"<<std::endl;
    CV_Assert(patchSize >= 2);

    bool do_keypoints = !useProvidedKeypoints;
    bool do_descriptors = _descriptors.needed();

    if( (!do_keypoints && !do_descriptors) || _image.empty() )
        return;

    //ROI handling
    const int HARRIS_BLOCK_SIZE = 9;
    int halfPatchSize = patchSize / 2;
    int border = std::max(edgeThreshold, std::max(halfPatchSize, HARRIS_BLOCK_SIZE/2))+1;

    Mat image = _image.getMat(), mask = _mask.getMat();
    if( image.type() != CV_8UC1 )
        cvtColor(_image, image, CV_BGR2GRAY);

    int levelsNum = this->nlevels;

    if( !do_keypoints )
    {
        // if we have pre-computed keypoints, they may use more levels than it is set in parameters
        // !!!TODO!!! implement more correct method, independent from the used keypoint detector.
        // Namely, the detector should provide correct size of each keypoint. Based on the keypoint size
        // and the algorithm used (i.e. BRIEF, running on 31x31 patches) we should compute the approximate
        // scale-factor that we need to apply. Then we should cluster all the computed scale-factors and
        // for each cluster compute the corresponding image.
        //
        // In short, ultimately the descriptor should
        // ignore octave parameter and deal only with the keypoint size.
        levelsNum = 0;
        for( size_t i = 0; i < _keypoints.size(); i++ )
            levelsNum = std::max(levelsNum, std::max(_keypoints[i].octave, 0));
        levelsNum++;
    }

    // Pre-compute the scale pyramids
    vector<Mat> imagePyramid(levelsNum), maskPyramid(levelsNum);
    for (int level = 0; level < levelsNum; ++level)
    {
        float scale = 1/getScale(level, firstLevel, scaleFactor);
        Size sz(cvRound(image.cols*scale), cvRound(image.rows*scale));
        Size wholeSize(sz.width + border*2, sz.height + border*2);
        Mat temp(wholeSize, image.type()), masktemp;
        imagePyramid[level] = temp(Rect(border, border, sz.width, sz.height));

        if( !mask.empty() )
        {
            masktemp = Mat(wholeSize, mask.type());
            maskPyramid[level] = masktemp(Rect(border, border, sz.width, sz.height));
        }

        // Compute the resized image
        if( level != firstLevel )
        {
            if( level < firstLevel )
            {
                resize(image, imagePyramid[level], sz, 0, 0, INTER_LINEAR);
                if (!mask.empty())
                    resize(mask, maskPyramid[level], sz, 0, 0, INTER_LINEAR);
            }
            else
            {
                resize(imagePyramid[level-1], imagePyramid[level], sz, 0, 0, INTER_LINEAR);
                if (!mask.empty())
                {
                    resize(maskPyramid[level-1], maskPyramid[level], sz, 0, 0, INTER_LINEAR);
                    threshold(maskPyramid[level], maskPyramid[level], 254, 0, THRESH_TOZERO);
                }
            }

            copyMakeBorder(imagePyramid[level], temp, border, border, border, border,
                           BORDER_REFLECT_101+BORDER_ISOLATED);
            if (!mask.empty())
                copyMakeBorder(maskPyramid[level], masktemp, border, border, border, border,
                               BORDER_CONSTANT+BORDER_ISOLATED);
        }
        else
        {
            copyMakeBorder(image, temp, border, border, border, border,
                           BORDER_REFLECT_101);
            if( !mask.empty() )
                copyMakeBorder(mask, masktemp, border, border, border, border,
                               BORDER_CONSTANT+BORDER_ISOLATED);
        }
    }

    // Pre-compute the keypoints (we keep the best over all scales, so this has to be done beforehand
    vector < vector<KeyPoint> > allKeypoints;
    if( do_keypoints )
    {
        // Get keypoints, those will be far enough from the border that no check will be required for the descriptor
        computeKeyPoints(imagePyramid, maskPyramid, allKeypoints,
                         nfeatures, firstLevel, scaleFactor,
                         edgeThreshold, patchSize, scoreType);

        // make sure we have the right number of keypoints keypoints
        /*vector<KeyPoint> temp;

        for (int level = 0; level < n_levels; ++level)
        {
            vector<KeyPoint>& keypoints = all_keypoints[level];
            temp.insert(temp.end(), keypoints.begin(), keypoints.end());
            keypoints.clear();
        }

        KeyPoint::retainBest(temp, n_features_);

        for (vector<KeyPoint>::iterator keypoint = temp.begin(),
             keypoint_end = temp.end(); keypoint != keypoint_end; ++keypoint)
            all_keypoints[keypoint->octave].push_back(*keypoint);*/
    }
    else
    {
        // Remove keypoints very close to the border
        KeyPointsFilter::runByImageBorder(_keypoints, image.size(), edgeThreshold);

        // Cluster the input keypoints depending on the level they were computed at
        allKeypoints.resize(levelsNum);
        for (vector<KeyPoint>::iterator keypoint = _keypoints.begin(),
             keypointEnd = _keypoints.end(); keypoint != keypointEnd; ++keypoint)
            allKeypoints[keypoint->octave].push_back(*keypoint);

        // Make sure we rescale the coordinates
        for (int level = 0; level < levelsNum; ++level)
        {
            if (level == firstLevel)
                continue;

            vector<KeyPoint> & keypoints = allKeypoints[level];
            float scale = 1/getScale(level, firstLevel, scaleFactor);
            for (vector<KeyPoint>::iterator keypoint = keypoints.begin(),
                 keypointEnd = keypoints.end(); keypoint != keypointEnd; ++keypoint)
                keypoint->pt *= scale;
        }
    }

    Mat descriptors;
    vector<Point> pattern;

    if( do_descriptors )
    {
        int nkeypoints = 0;
        for (int level = 0; level < levelsNum; ++level)
            nkeypoints += (int)allKeypoints[level].size();
        if( nkeypoints == 0 )
            _descriptors.release();
        else
        {
            _descriptors.create(nkeypoints, descriptorSize(), CV_8U);
            descriptors = _descriptors.getMat();
        }

        const int npoints = 512;
        Point patternbuf[npoints];
        const Point* pattern0 = (const Point*)bit_pattern_31_;

        if( patchSize != 31 )
        {
            pattern0 = patternbuf;
            makeRandomPattern(patchSize, patternbuf, npoints);
        }

        CV_Assert( WTA_K == 2 || WTA_K == 3 || WTA_K == 4 );

        if( WTA_K == 2 )
            std::copy(pattern0, pattern0 + npoints, std::back_inserter(pattern));
        else
        {
            int ntuples = descriptorSize()*4;
            initializeORB2Pattern(pattern0, pattern, ntuples, WTA_K, npoints);
        }
    }

    _keypoints.clear();
    int offset = 0;
    for (int level = 0; level < levelsNum; ++level)
    {
        // Get the features and compute their orientation
        vector<KeyPoint>& keypoints = allKeypoints[level];
        int nkeypoints = (int)keypoints.size();

        // Compute the descriptors
        if (do_descriptors)
        {
            Mat desc;
            if (!descriptors.empty())
            {
                desc = descriptors.rowRange(offset, offset + nkeypoints);
            }

            offset += nkeypoints;
            // preprocess the resized image
            Mat& workingMat = imagePyramid[level];
            //boxFilter(workingMat, workingMat, workingMat.depth(), Size(5,5), Point(-1,-1), true, BORDER_REFLECT_101);
            GaussianBlur2(workingMat, workingMat, Size(7, 7), 2, 2, BORDER_REFLECT_101);
            computeDescriptors(workingMat, keypoints, desc, pattern, descriptorSize(), WTA_K);
        }

        // Copy to the output data
        if (level != firstLevel)
        {
            float scale = getScale(level, firstLevel, scaleFactor);
            for (vector<KeyPoint>::iterator keypoint = keypoints.begin(),
                 keypointEnd = keypoints.end(); keypoint != keypointEnd; ++keypoint)
                keypoint->pt *= scale;
        }
        // And add the keypoints to the output
        _keypoints.insert(_keypoints.end(), keypoints.begin(), keypoints.end());
    }
}

void ORB2::detectImpl( const Mat& image, vector<KeyPoint>& keypoints, const Mat& mask) const
{
    (*this).my_operator(image, mask, keypoints, noArray(), false);
}

void ORB2::computeImpl( const Mat& image, vector<KeyPoint>& keypoints, Mat& descriptors) const
{
    (*this).my_operator(image, Mat(), keypoints, descriptors, true);
}
void readme();

/**
 * @function main
 * @brief Main function
 */
int main( int argc, char** argv )
{
  if( argc != 3 )
  { return -1; }

  Mat img_1 = imread( argv[1], CV_LOAD_IMAGE_GRAYSCALE );
  Mat img_2 = imread( argv[2], CV_LOAD_IMAGE_GRAYSCALE );

  if( !img_1.data || !img_2.data )
  { return -1; }

  //-- Step 1: Detect the keypoints using SURF Detector
  int minHessian = 400;

  ORB2 detector( minHessian );

  std::vector<KeyPoint> keypoints_1, keypoints_2;

  detector.detectImpl( img_1, keypoints_1 );
  printf("keypoints_1 - %d \n",(int) keypoints_1.size());
  detector.detectImpl( img_2, keypoints_2 );
  printf("keypoints_2 - %d \n",(int) keypoints_2.size());

  //-- Step 2: Calculate descriptors (feature vectors)
  ORB2 extractor;

  Mat descriptors_1, descriptors_2;

  extractor.computeImpl( img_1, keypoints_1, descriptors_1 );
  //printf("descriptor_1 - %d \n",descriptors_1.size());
  extractor.computeImpl( img_2, keypoints_2, descriptors_2 );
  //printf("descriptor_2 - %d \n",descriptors_2.size());

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
{ std::cout << " Usage: ./SURF_descriptor <img1> <img2>" << std::endl; }

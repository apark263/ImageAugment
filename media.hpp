#include <stdexcept>
#include <random>
#include <chrono>
using namespace std;
using namespace cv;
typedef Rect_<float> Rectf;

enum MediaType {
    IMAGE = 0,
    VIDEO = 1,
    AUDIO = 2,
    TEXT = 3,
};

class MediaParams {
public:
    MediaParams(int mtype) : _mtype(mtype) {
    }

    // Do not make this virtual. The object passed down from Python will not
    // have the virtual function table filled in.
    void dump() {
        printf("mtype %d\n", _mtype);
    }

public:
    int                         _mtype;
};

float CPCA[3][3] = {{0.39731118,  0.70119634, -0.59200296},
                    {-0.81698062, -0.02354167, -0.5761844},
                    {0.41795513, -0.71257945, -0.56351045}};

float CSTD[3][3] = {{19.72083305, 0, 0},
                    {0, 37.09388853, 0},
                    {0, 0, 121.78006099}};

float GSCL[3][3] = {{0.114, 0.587, 0.299},
                    {0.114, 0.587, 0.299},
                    {0.114, 0.587, 0.299}};



class ImageParams : public MediaParams {
public:
    ImageParams()
    : MediaParams(IMAGE) {
        _augParams = {{"channelCount", 3}, {"height", 224}, {"width", 224},
                      {"cropRange", 0}, {"doFlip", 0},
                      {"minScale", 100}, {"minAspectRatio", 100},
                      {"contrastRange", 0}, {"brightnessRange", 0},
                      {"angleRange", 0}, {"fixedScale", 0},
                      {"R_mean", 104}, {"G_mean", 119}, {"B_mean", 127},
                      {"matchAspectRatio", 0}};
        Mat cpca(3, 3, CV_32FC1, CPCA);
        Mat cstd(3, 3, CV_32FC1, CSTD);
        cpca *= cstd;
        // _lgt = new Mat(3, 3, CV_32FC1, CPCA);
        // Mat aa(3, 3, CV_32FC1, CSTD);
        // (*_lgt) *= aa;
    }

    ImageParams(vector<string> keys, vector<int> vals)
    : ImageParams() {
        set_keys(keys, vals);
    }

    void set_keys(int npairs, char **keys, int *vals) {
        for (int i=0; i < npairs; i++) {
            set_key(string(keys[i]), vals[i]);
        }
    }

    void set_keys(vector<string> keys, vector<int> vals) {
        for (size_t i = 0; i < keys.size(); i++) {
            set_key(keys[i], vals[i]);
        }
    }

    void set_keys(map<string, int> kvmap) {
        for (auto x: kvmap) {
            set_key(x.first, x.second);
        }
    }

    void set_key(string key, int val) {
        auto vv = _augParams.find(key);
        if (vv == _augParams.end()) {
            throw std::runtime_error("Gak! invalid key: " + key);
        } else {
            // Should check for invalid params
            vv->second = val;
        }
    }

    void dump() {
        MediaParams::dump();
        for (auto x: _augParams) {
            cout << x.first << ": " << x.second << endl;
        }
    }

    int& operator [] (string b) {
        return _augParams.at(b);
    }

public:
    map<string, int>            _augParams;
    Mat* _lgt;
};


class Image {
public:
    Image(ImageParams &ip)
    : _outSize(ip["width"], ip["height"]),
      _ip_minAR(ip["minAspectRatio"] / 100.0f),
      _ip_minScale(ip["minScale"] / 100.0f),
      _ip_contrastRange(ip["contrastRange"] / 100.0f),
      _ip_brightnessRange(ip["brightnessRange"] / 100.0f),
      _ip_angleRange(ip["angleRange"]),
      _ip_cropRange(ip["cropRange"] / 100.0f),
      _ip_fixedScale(ip["fixedScale"]),
      _ip_matchAR(ip["matchAspectRatio"]),
      _ip_doFlip(ip["doFlip"])
  {
        unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();

        default_random_engine generator(seed1);
        uniform_real_distribution<float> distribution(0.0, 1.0);
        _rng = std::bind(distribution, generator);
    }

    float urand(float low, float high) {
        return _rng() * (high - low) + low;
    }

    float urand(float high) {
        return urand(0.0, high);
    }

    float urand_zcent(float high) {
        return urand(-high, high);
    }

    int urand_binary() {
        return _rng() > 0.5 ? 1 : 0;
    }

    void decode(char* itemBuf, int itemSize, char* outBuf) {
        Mat decodedImage = cv::imdecode(Mat(1, itemSize, CV_8UC3, itemBuf), CV_LOAD_IMAGE_COLOR);
        Mat transformedImage;
        transform(decodedImage, transformedImage);

        int offset = (int) _outSize.area();
        Mat ch_b(_outSize, CV_8U, outBuf + offset*0);
        Mat ch_g(_outSize, CV_8U, outBuf + offset*1);
        Mat ch_r(_outSize, CV_8U, outBuf + offset*2);
        Mat channels[3] = {ch_b, ch_g, ch_r};

        cv::split(transformedImage, channels);
    }

    // This function sets the transform params according to the ranges given by ImageParams
    void transform(Mat &inMat, Mat &outMat) {
        Size2f inSize = inMat.size();
        Mat rotatedImg, croppedImg, flippedImg;

        /*************
        *  CROPPING  *
        **************/
        Rectf cropBox;
        getCropBox(inSize, cropBox);
        croppedImg = inMat(cropBox).clone();

        /*************
        *  FLIPPING  *
        **************/
        if (_ip_doFlip && urand_binary()) {
            flip(croppedImg, flippedImg, 1);
        } else {
            flippedImg = croppedImg;
        }
        bool doEnlarge = flippedImg.size().area() / _outSize.area() > 1.0;

        /**************************
        *  LIGHTING PERTURBATION  *
        ***************************/

        float pxnoise[3];
        pxnoise[0] = urand_zcent(0.5);
        pxnoise[1] = urand_zcent(0.5);
        pxnoise[2] = urand_zcent(0.5);
        const Mat cpca(3, 3, CV_32FC1, CPCA);
        Mat alphas(3, 1, CV_32FC1, pxnoise);
        alphas = cpca * alphas;
        Mat pixel = alphas.reshape(3, 1);
        flippedImg = (flippedImg + pixel.at<Scalar_<float>>(0, 0));

        // Brightness and saturation
        float satbuf[3][3];
        memcpy(satbuf, GSCL, 9 * sizeof(float));
        Mat satmtx(3, 3, CV_32FC1, satbuf);
        float st_alpha = urand_zcent(0.4);
        // float br_alpha = urand_zcent(0.4) + 1;
        // satmtx = br_alpha * ((Mat::eye(3, 3, CV_32FC1) - satmtx) * st_alpha + satmtx);
        satmtx = (st_alpha * Mat::eye(3, 3, CV_32FC1) + (1.0 - st_alpha) * satmtx);
        Mat saturated;
        cv::transform(flippedImg, saturated, satmtx);
        cout << st_alpha << " " << satmtx << endl;
        // Mat smat = flippedImg.reshape(1, flippedImg.rows * flippedImg.cols);
        // smat = smat.t() * satmtx.t();
        // float contrast = urand_zcent(_ip_contrastRange);
        // float brightness = urand_zcent(_ip_brightnessRange);
        // if (contrast != 0 || brightness != 0) {
        //     flippedImg.convertTo(flippedImg, -1, 1.0f + contrast, 127.0 * brightness);
        // }

        /*************
        *  RESIZING  *
        *************/
        resize(saturated, outMat, _outSize, 0, 0, doEnlarge ? CV_INTER_AREA : CV_INTER_CUBIC);
    }

    void getCropBox(Size2f inSize, Rectf &cropBox) {
        float scale, origAR, cropAR;
        origAR = inSize.width / inSize.height;
        cropAR = _ip_matchAR ? origAR : urand(_ip_minAR, 1.0f / _ip_minAR);
        bool portrait = origAR < 1;

        if (_ip_fixedScale != 0) {
            // This is the case where we want to take a crop of a fixed rescaling
            scale = min(_outSize.width, _outSize.height) / _ip_fixedScale;
        } else {
            float maxScale = cropAR > origAR ? origAR / cropAR : cropAR / origAR;
            scale = sqrt(urand(min(_ip_minScale, maxScale), maxScale));
        }

        Vec2f c_sz(inSize.width, inSize.height);
        c_sz[portrait] = c_sz[!portrait] * (portrait ? 1 / cropAR : cropAR);
        c_sz *= scale;

        // This is the border size (and center offset)
        Vec2f c_xy(inSize.width, inSize.height);
        c_xy = (c_xy - c_sz) / 2.0;
        Vec2f c_offset(urand_zcent(c_xy[0]), urand_zcent(c_xy[1]));
        c_xy += c_offset * _ip_cropRange;

        cropBox = Rectf((Point2f) c_xy, (Size2f) c_sz);
    }

    void dump() {
        cout << "_outSize " << _outSize << endl;
        cout << "_ip_minAR " << _ip_minAR << endl;
        cout << "_ip_minScale " << _ip_minScale << endl;
        cout << "_ip_contrastRange " << _ip_contrastRange << endl;
        cout << "_ip_brightnessRange " << _ip_brightnessRange << endl;
        cout << "_ip_angleRange " << _ip_angleRange << endl;
        cout << "_ip_cropRange " << _ip_cropRange << endl;
        cout << "_ip_fixedScale " << _ip_fixedScale << endl;
        cout << "_ip_matchAR " << _ip_matchAR << endl;
        cout << "_ip_doFlip " << _ip_doFlip << endl;
    }

public:
    Size2f          _outSize;

    // These are parameters that define what type of augmentation is done
    float _ip_minAR;
    float _ip_minScale;
    float _ip_contrastRange;
    float _ip_brightnessRange;
    float _ip_angleRange;
    float _ip_cropRange;
    int   _ip_fixedScale;
    int   _ip_matchAR;
    int   _ip_doFlip;
    std::function<float()> _rng;
};


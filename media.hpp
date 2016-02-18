#include <stdexcept>
#include <random>

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


class ImageParams : public MediaParams {
public:
    ImageParams()
    : MediaParams(IMAGE) {
        _augParams = {{"channelCount", 3}, {"height", 224}, {"width", 224},
                      {"cropRange", 0}, {"flipRange", 1},
                      {"minScale", 100}, {"minAspectRatio", 100},
                      {"contrastRange", 0}, {"brightnessRange", 0},
                      {"angleRange", 0}, {"fixedScale", 0},
                      {"R_mean", 104}, {"G_mean", 119}, {"B_mean", 127},
                      {"matchAspectRatio", 0}};
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
};


class Image {
public:
    Image(ImageParams &ip)
    : _outSize(ip["width"], ip["height"]), _flip(0),
    _contrast(0), _brightness(0), _angle(0), _cropbox(Rectf()),
    _ip_minAR(1.0), _ip_minScale(1.0), _ip_contrastRange(0.0),
    _ip_brightnessRange(0.0), _ip_angleRange(0.0), _ip_cropRange(0.0),
    _ip_fixedScale(0), _ip_matchAR(0), _ip_flipRange(1)
  {
        _ip_minAR           = ip["minAspectRatio"] / 100.0f;
        _ip_minScale        = ip["minScale"] / 100.0f;
        _ip_contrastRange   = ip["contrastRange"] / 100.0f;
        _ip_brightnessRange = ip["brightnessRange"] / 100.0f;
        _ip_angleRange      = ip["angleRange"];
        _ip_cropRange       = ip["cropRange"] / 100.0f;
        _ip_fixedScale      = ip["fixedScale"];
        _ip_matchAR         = ip["matchAspectRatio"];
        _ip_flipRange       = ip["flipRange"];
        // Scalar pixel_mean(ip["B_mean"], ip["G_mean"], ip["R_mean"]);
        default_random_engine generator;
        uniform_real_distribution<float> distribution(0.0, 1.0);
        _rng = std::bind(distribution, generator);
    }

    // This function sets the transform params according to the ranges given by ImageParams
    void randomize(RNG &rng, Size2f &inSize) {
        _flip       = rng.uniform(0, _ip_flipRange);
        _contrast   = rng.uniform(-_ip_contrastRange, _ip_contrastRange);
        _brightness = rng.uniform(-_ip_brightnessRange, _ip_brightnessRange);
        _angle      = rng.uniform(-_ip_angleRange, _ip_angleRange);

        // Now do the cropbox
        float scale, origAR, cropAR;
        origAR = inSize.width / inSize.height;
        cropAR = _ip_matchAR ? origAR : rng.uniform(_ip_minAR, 1.0f / _ip_minAR);
        bool portrait = origAR < 1;

        if (_ip_fixedScale != 0) {
            // This is the case where we want to take a crop of a fixed rescaling
            scale = min(_outSize.width, _outSize.height) / _ip_fixedScale;
        } else {
            float maxScale = cropAR > origAR ? origAR / cropAR : cropAR / origAR;
            scale = sqrt(rng.uniform(min(_ip_minScale, maxScale), maxScale));
        }

        Vec2f c_sz(inSize.width, inSize.height);
        Vec2f c_xy = c_sz;
        c_sz[portrait] = c_sz[!portrait] * (portrait ? 1 / cropAR : cropAR);
        c_sz *= scale;

        // This is the border size (and center offset)
        c_xy = (c_xy - c_sz) / 2.0;
        Vec2f c_offset(rng.uniform(-c_xy[0], c_xy[0]), rng.uniform(-c_xy[1], c_xy[1]));
        c_xy += c_offset * _ip_cropRange;

        _cropbox = Rectf((Point2f) c_xy, (Size2f) c_sz);
    }

    void flipImage(Mat &inMat, Mat &outMat) {
        if (_flip != 0) {
            flip(inMat, outMat, 1);
        } else {
            outMat = inMat;
        }
    }

    void cropImage(Mat &inMat, Mat &outMat) {
        outMat = inMat(_cropbox);
    }

    void adjustImage(Mat &inMat) {
        if (_contrast != 0 || _brightness != 0) {
            inMat.convertTo(inMat, -1, 1.0f + _contrast, 127.0 * _brightness);
        }
    }

    void rotateImage(Mat &inMat, Mat &outMat) {
        if (_angle == 0.0f) {
            outMat = inMat;
        } else {
            warpAffine( inMat, outMat,
                        getRotationMatrix2D( Point2f(inMat.size()) * 0.5f, _angle, 1.0f ),
                        inMat.size() );
        }
    }

    void resizeImage(Mat &inMat, Mat &outMat) {
        int interp_method = inMat.size().area() < _outSize.area() ? CV_INTER_AREA : CV_INTER_CUBIC;
        resize(inMat, outMat, _outSize, 0, 0, interp_method);
    }

    void transform(Mat &inMat, Mat &outMat) {
        Mat rotatedImg, croppedImg, flippedImg;
        rotateImage(inMat, rotatedImg);
        cropImage(rotatedImg, croppedImg);
        flipImage(croppedImg, flippedImg);
        adjustImage(flippedImg);
        resizeImage(flippedImg, outMat);
    }

    void dump() {
         cout << "outSize " << _outSize << endl;
         cout << "flip " << _flip << endl;
         cout << "contrast " << _contrast << endl;
         cout << "brightness " << _brightness << endl;
         cout << "angle " << _angle << endl;
         cout << "cropbox " << _cropbox << endl;
    }

public:
    Size2f          _outSize;
    int             _flip;
    float           _contrast;
    float           _brightness;
    float           _angle;
    Rectf           _cropbox;

    // These are parameters that define what type of augmentation is done
    float _ip_minAR;
    float _ip_minScale;
    float _ip_contrastRange;
    float _ip_brightnessRange;
    float _ip_angleRange;
    float _ip_cropRange;
    int   _ip_fixedScale;
    int   _ip_matchAR;
    int   _ip_flipRange;
    bind<uniform_real_distribution<float> &, default_random_engine> _rng;
};


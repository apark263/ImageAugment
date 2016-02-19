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
                      {"cropRange", 0}, {"doFlip", 0},
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
        default_random_engine generator;
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

    // This function sets the transform params according to the ranges given by ImageParams
    void transform(Mat &inMat, Mat &outMat) {
        Size2f inSize = inMat.size();
        Mat rotatedImg, croppedImg, flippedImg;

        // Rotation
        float angle = urand_zcent(_ip_angleRange);
        if (angle != 0.0f) {
            warpAffine( inMat, rotatedImg,
                        getRotationMatrix2D( Point2f(inSize) * 0.5f, angle, 1.0f ),
                        inSize );
        } else {
            rotatedImg = inMat;
        }

        // Now do the cropbox
        Rectf cropBox;
        getCropBox(inSize, cropBox);
        croppedImg = rotatedImg(cropBox);

        // Flip
        if (_ip_doFlip && urand_binary()) {
            flip(croppedImg, flippedImg, 1);
        } else {
            flippedImg = croppedImg;
        }

        // contrast and brightness adjustment
        float contrast = urand_zcent(_ip_contrastRange);
        float brightness = urand_zcent(_ip_brightnessRange);
        if (contrast != 0 || brightness != 0) {
            flippedImg.convertTo(flippedImg, -1, 1.0f + contrast, 127.0 * brightness);
        }

        // Resize into final output
        bool doEnlarge = flippedImg.size().area() / _outSize.area() > 1.0;
        resize(flippedImg, outMat, _outSize, 0, 0, doEnlarge ? CV_INTER_AREA : CV_INTER_CUBIC);
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


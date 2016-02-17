#include <stdexcept>

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
                      {"angleRange", 0}, {"fixedScale", -1}};
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


class ImageTransform {
public:
    ImageTransform(Size2f outSize)
    : _outSize(outSize), _flip(0),
    _contrast(0), _brightness(0), _angle(0), _cropbox(Rectf())  {

    }

    // This function sets the transform params according to the ranges given by ImageParams
    void randomize(RNG &rng, Size2f &inSize, ImageParams &ip) {
        float minAspectRatio  = ip["minAspectRatio"] / 100.0f;
        float minScale        = ip["minScale"] / 100.0f;
        float contrastRange   = ip["contrastRange"] / 100.0f;
        float brightnessRange = ip["brightnessRange"] / 100.0f;
        float angleRange      = ip["angleRange"];
        float cropRange       = ip["cropRange"] / 100.0f;
        int fixedScale        = ip["fixedScale"];

        _flip = rng.uniform(0, ip["flipRange"]);
        _contrast = rng.uniform(-contrastRange, contrastRange);
        _brightness = rng.uniform(-brightnessRange, brightnessRange);
        _angle = rng.uniform(-angleRange, angleRange);

        // Now do the cropbox
        float oAR = inSize.width / inSize.height;
        float cAR = rng.uniform(minAspectRatio, 1.0f / minAspectRatio);
        float scale;
        if (fixedScale < 0) {
            float maxScale = cAR > oAR ? oAR / cAR : cAR / oAR;
            scale = sqrt(rng.uniform(min(minScale, maxScale), maxScale));
        } else {
            // Math here is wrong
            scale = min(outSize.width, outSize.height) / fixedScale;
        }


        if (oAR < 1) {
            _cropbox.width = inSize.width * scale;
            _cropbox.height = _cropbox.width / cAR;
        } else {
            _cropbox.height = inSize.height * scale;
            _cropbox.width = _cropbox.height * cAR;
        }

        Size2f border = inSize - _cropbox.size();
        _cropbox.x = border.width / 2;
        _cropbox.y = border.height / 2;
        border.width = _cropbox.x * cropRange;
        border.height = _cropbox.y * cropRange;

        _cropbox.x += rng.uniform(-border.width, border.width);
        _cropbox.y += rng.uniform(-border.height, border.height);
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
};


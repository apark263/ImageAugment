
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
    ImageParams(int channelCount, int height, int width,
                bool center, int flipRange,
                float minScale, float minAspectRatio,
                float contrastRange, float brightnessRange, float angleRange)
    : MediaParams(IMAGE),
     _channelCount(channelCount), _height(height), _width(width),
     _center(center), _flipRange(flipRange),
     _minScale(minScale), _minAspectRatio(minAspectRatio),
     _contrastRange(contrastRange), _brightnessRange(brightnessRange),
     _angleRange(angleRange) {
    }

    ImageParams()
    : ImageParams(3, 224, 224, true, 1, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f) {}

    void dump() {
        MediaParams::dump();
    }


public:
    int                         _channelCount;
    int                         _height;
    int                         _width;
    bool                        _center;
    int                         _flipRange;
    float                       _minScale;
    float                       _minAspectRatio;
    float                       _contrastRange;
    float                       _brightnessRange;
    float                       _angleRange;
};


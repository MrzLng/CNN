#pragma once
#include <string>

using namespace std;

class Image
{
    public:
    Image(string path);
    ~Image();

    inline unsigned char* getDataPtr() const {return data;}
    inline unsigned char getData(int index) const {return data[index];}
    inline unsigned char getData(int x, int y, int channel) const {return data[y*width*channels+x*channels+channel];}
    inline int getWidth() const {return width;}
    inline int getHeight() const {return height;}
    inline int getChannels() const {return channels;}

    private:
    string path;
    unsigned char* data;
    int width;
    int height;
    int channels;
};
#include <algorithm>
#include <iostream>

#include "buffer.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

Buffer::Buffer(int width, int height, int channels):width(width), height(height), channels(channels), data(width*height*channels), size(width*height*channels) {}

Buffer::Buffer(const Image& image):width(image.getWidth()), height(image.getHeight()), channels(3), data(image.getWidth()*image.getHeight()*3), size(image.getWidth()*image.getHeight()*3)
{
    for (int y=0; y<height; y++)
    {
        for (int x=0; x<width; x++)
        {
            setData(static_cast<float>(image.getData(x,y,0))/255.0f,x,y,0);
            setData(static_cast<float>(image.getData(x,y,1))/255.0f,x,y,1);
            setData(static_cast<float>(image.getData(x,y,2))/255.0f,x,y,2);
        }
    }
}

Buffer Buffer::padding(Padding_Type paddingType, int pad) const
{
    Buffer result(width+2*pad, height+2*pad, channels);

    int y=0;
    switch(paddingType)
    {
        case Padding_Type::ZERO_PADDING:
        fill(result.getVector(), result.getVector()+result.getSize(), 0);
        for (int y=pad; y<height+pad; y++)
        {
            for (int x=pad; x<width+pad; x++)
            {
                result.setData(getData(x-pad,y-pad,0),x,y,0);
                result.setData(getData(x-pad,y-pad,1),x,y,1);
                result.setData(getData(x-pad,y-pad,2),x,y,2);
            }
        }
        break;
        case Padding_Type::EXTEND_EDGE:
        for (;y<pad;y++)
        { 
            int x=0;
            for (;x<pad;x++)
            {
                result.setData(getData(0, 0, 0), x, y, 0);
                result.setData(getData(0, 0, 1), x, y, 1);
                result.setData(getData(0, 0, 2), x, y, 2);
            }
            for (;x<getWidth()+pad;x++)
            {
                result.setData(getData(x-pad, 0, 0), x, y, 0);
                result.setData(getData(x-pad, 0, 1), x, y, 1);
                result.setData(getData(x-pad, 0, 2), x, y, 2);
            }
            for (;x<getWidth()+2*pad;x++)
            {
                result.setData(getData(getWidth()-1, 0, 0), x, y, 0);
                result.setData(getData(getWidth()-1, 0, 1), x, y, 1);
                result.setData(getData(getWidth()-1, 0, 2), x, y, 2);
            }
        }
        for (;y<getHeight()+pad;y++)
        {
            int x=0;
            for (;x<pad;x++)
            {
                result.setData(getData(0, y-pad, 0), x, y, 0);
                result.setData(getData(0, y-pad, 1), x, y, 1);
                result.setData(getData(0, y-pad, 2), x, y, 2);
            }
            for (;x<getWidth()+pad;x++)
            {
                result.setData(getData(x-pad, y-pad, 0), x, y, 0);
                result.setData(getData(x-pad, y-pad, 1), x, y, 1);
                result.setData(getData(x-pad, y-pad, 2), x, y, 2);
            }
            for (;x<getWidth()+2*pad;x++)
            {
                result.setData(getData(getWidth()-1, y-pad, 0), x, y, 0);
                result.setData(getData(getWidth()-1, y-pad, 1), x, y, 1);
                result.setData(getData(getWidth()-1, y-pad, 2), x, y, 2);
            }
        }
        for (;y<getHeight()+2*pad;y++)
        {
            int x=0;
            for (;x<pad;x++)
            {
                result.setData(getData(0, getHeight()-1, 0), x, y, 0);
                result.setData(getData(0, getHeight()-1, 1), x, y, 1);
                result.setData(getData(0, getHeight()-1, 2), x, y, 2);
            }
            for (;x<getWidth()+pad;x++)
            {
                result.setData(getData(x-pad, getHeight()-1, 0), x, y, 0);
                result.setData(getData(x-pad, getHeight()-1, 1), x, y, 1);
                result.setData(getData(x-pad, getHeight()-1, 2), x, y, 2);
            }
            for (;x<getWidth()+2*pad;x++)
            {
                result.setData(getData(getWidth()-1, getHeight()-1, 0), x, y, 0);
                result.setData(getData(getWidth()-1, getHeight()-1, 1), x, y, 1);
                result.setData(getData(getWidth()-1, getHeight()-1, 2), x, y, 2);
            }
        }
        break;
        case Padding_Type::REFLECTION:
        for (;y<pad;y++)
        { 
            int x=0;
            for (;x<pad;x++)
            {
                result.setData(getData(pad-x-1,pad-y-1, 0), x, y, 0);
                result.setData(getData(pad-x-1,pad-y-1, 1), x, y, 1);
                result.setData(getData(pad-x-1,pad-y-1, 2), x, y, 2);
            }
            for (;x<getWidth()+pad;x++)
            {
                result.setData(getData(x-pad, pad-y-1, 0), x, y, 0);
                result.setData(getData(x-pad, pad-y-1, 1), x, y, 1);
                result.setData(getData(x-pad, pad-y-1, 2), x, y, 2);
            }
            for (;x<getWidth()+2*pad;x++)
            {
                result.setData(getData(getWidth()*2-x+pad-1, pad-y-1, 0), x, y, 0);
                result.setData(getData(getWidth()*2-x+pad-1, pad-y-1, 1), x, y, 1);
                result.setData(getData(getWidth()*2-x+pad-1, pad-y-1, 2), x, y, 2);
            }
        }
        for (;y<getHeight()+pad;y++)
        {
            int x=0;
            for (;x<pad;x++)
            {
                result.setData(getData(pad-x-1, y-pad, 0), x, y, 0);
                result.setData(getData(pad-x-1, y-pad, 1), x, y, 1);
                result.setData(getData(pad-x-1, y-pad, 2), x, y, 2);
            }
            for (;x<getWidth()+pad;x++)
            {
                result.setData(getData(x-pad, y-pad, 0), x, y, 0);
                result.setData(getData(x-pad, y-pad, 1), x, y, 1);
                result.setData(getData(x-pad, y-pad, 2), x, y, 2);
            }
            for (;x<getWidth()+2*pad;x++)
            {
                result.setData(getData(getWidth()*2-x+pad-1, y-pad, 0), x, y, 0);
                result.setData(getData(getWidth()*2-x+pad-1, y-pad, 1), x, y, 1);
                result.setData(getData(getWidth()*2-x+pad-1, y-pad, 2), x, y, 2);
            }
        }
        for (;y<getHeight()+2*pad;y++)
        {
            int x=0;
            for (;x<pad;x++)
            {
                result.setData(getData(pad-x-1, getHeight()*2-y+pad-1, 0), x, y, 0);
                result.setData(getData(pad-x-1, getHeight()*2-y+pad-1, 1), x, y, 1);
                result.setData(getData(pad-x-1, getHeight()*2-y+pad-1, 2), x, y, 2);
            }
            for (;x<getWidth()+pad;x++)
            {
                result.setData(getData(x-pad, getHeight()*2-y+pad-1, 0), x, y, 0);
                result.setData(getData(x-pad, getHeight()*2-y+pad-1, 1), x, y, 1);
                result.setData(getData(x-pad, getHeight()*2-y+pad-1, 2), x, y, 2);
            }
            for (;x<getWidth()+2*pad;x++)
            {
                result.setData(getData(getWidth()*2-x+pad-1, getHeight()*2-y+pad-1, 0), x, y, 0);
                result.setData(getData(getWidth()*2-x+pad-1, getHeight()*2-y+pad-1, 0), x, y, 1);
                result.setData(getData(getWidth()*2-x+pad-1, getHeight()*2-y+pad-1, 0), x, y, 2);
            }
        }
        break;
    }
    return result;
}

void Buffer::toImage(string path)
{
    unsigned char* finalImage = new unsigned char[width * height * 3];
    for (int y=0; y<height; y++)
    {
        for (int x=0; x<width; x++)
        {
            finalImage[y*width*3+x*3+0] = static_cast<unsigned char>(clamp(getData(x,y,0)*255.0f, 0.0f, 255.0f));
            finalImage[y*width*3+x*3+1] = static_cast<unsigned char>(clamp(getData(x,y,1)*255.0f, 0.0f, 255.0f));
            finalImage[y*width*3+x*3+2] = static_cast<unsigned char>(clamp(getData(x,y,2)*255.0f, 0.0f, 255.0f));
        }
    }

    stbi_write_png(path.c_str(), width, height, 3, finalImage, width*3);
    delete[] finalImage;
}
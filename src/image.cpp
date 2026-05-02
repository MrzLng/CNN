#include "image.h"

#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

Image::Image(string path):path(path)
{
    data = stbi_load(path.c_str(), &width, &height, &channels, 3);
    if (!data)
    {
        const char* reason = stbi_failure_reason();
        cerr << "[Failed to load image]: " << reason << endl;
    }
}

Image::~Image()
{
    stbi_image_free(data);
}
#pragma once

#include "kernel.h"

#include <string>
#include <vector>
using namespace std;

class KernelBank
{
    public:
    KernelBank(int width, int height, int channels, int count);
    KernelBank(string path);

    inline int getWidth() {return width;}
    inline int getHeight() {return height;}
    inline int getChannels() {return channels;}
    inline int getCount() {return count;}
    inline Kernel& getKernel(int n) {return kernels[n];}

    void write(string path);
    void print();

    private:
    int width;
    int height;
    int channels;
    int count;
    vector<Kernel> kernels;
};
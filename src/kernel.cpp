#include "kernel.h"

#include <string>
#include <fstream>
#include <iostream>

Kernel::Kernel(std::string path)
{
    std::ifstream file(path, std::ios::binary);

    if (!file.is_open())
    {
        std::cerr << "could not open the file" << std::endl;
        return;
    }

    file.read(reinterpret_cast<char *>(&this->width), sizeof(int));
    file.read(reinterpret_cast<char *>(&this->height), sizeof(int));
    file.read(reinterpret_cast<char *>(&this->channels), sizeof(int));
    data.resize(this->width * this->height * (this->channels != 0 ? this->channels : 1));
    file.read(reinterpret_cast<char *>(this->data.data()), sizeof(float) * this->width * this->height * (this->channels != 0 ? this->channels : 1));
}

void Kernel::write(std::string path)
{
    std::ofstream file(path, std::ios::binary);

    if (!file)
    {
        std::cerr << "could not open the file" << std::endl;
        return;
    }

    file.write(reinterpret_cast<char *>(&this->width), sizeof(int));
    file.write(reinterpret_cast<char *>(&this->height), sizeof(int));
    file.write(reinterpret_cast<char *>(&this->channels), sizeof(int));
    file.write(reinterpret_cast<char *>(this->data.data()), sizeof(float) * this->width * this->height * (this->channels != 0 ? this->channels : 1));
}
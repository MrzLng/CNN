#pragma once

#include <ctime>
#include <string>

using namespace std;

class Timer
{
    public:
    Timer(string functionName);
    ~Timer();

    private:
    string functionName;
    clock_t startTime;
};
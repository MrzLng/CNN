#include "timer.h"

#include <iostream>

using namespace std;

Timer::Timer(string functionName): functionName(functionName)
{
    startTime = clock();
}

Timer::~Timer()
{
    cout << functionName << ": " << double(clock()-startTime)/CLOCKS_PER_SEC << " seconds elapsed" << endl;
}
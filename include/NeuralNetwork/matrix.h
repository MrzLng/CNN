#pragma once

#include <vector>
#include <string>

using namespace std;

class Matrix
{
    public:

    Matrix(vector<float> data, int rows, int columns);
    Matrix(float value, int rows, int columns);
    Matrix(int rows, int columns);
    Matrix(string path);

    inline int getRows() const {return rows;}
    inline int getColumns() const {return columns;}
    inline float getData(int row, int column) const {return data[row*columns+column];}
    inline float& getData(int row, int column) {return data[row*columns+column];}

    void write(string path);

    void resize(int rows, int columns);

    inline float* operator[](int row) {return &data[row*columns];}
    inline const float* operator[](int row) const {return &data[row*columns];}

    Matrix operator+(const Matrix& other) const;
    Matrix operator-(const Matrix& other) const;
    Matrix operator*(const Matrix& other) const;

    Matrix& operator+=(const Matrix& other);
    Matrix& operator-=(const Matrix& other);
    Matrix& operator*=(const Matrix& other);

    private:

    vector<float> data;
    int rows;
    int columns;

};
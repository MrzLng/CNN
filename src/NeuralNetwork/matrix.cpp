#include "matrix.h"

#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <omp.h>
#include <random>
#include <algorithm>

using namespace std;

Matrix::Matrix(vector<float> data, int rows, int columns)
    : data(data), rows(rows), columns(columns)
{
    if (data.size() != rows * columns)
    {
        cerr << "matrix dimensions does not match data" << endl;
    }
}

Matrix::Matrix(float value, int rows, int columns)
    : data(rows * columns, value), rows(rows), columns(columns) {}

Matrix::Matrix(int rows, int columns)
    : data(rows * columns), rows(rows), columns(columns) {}

Matrix::Matrix(int type, float arg1, float arg2, int rows, int columns)
    : data(rows * columns), rows(rows), columns(columns)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    switch (type)
    {
    case RandomType::normal:
    {
        std::normal_distribution<float> ndistr(arg1, arg2);
        for (int i = 0; i < rows * columns; i++)
        {
            data[i] = ndistr(gen);
        }
        break;
    }
    case RandomType::standard:
    {
        std::uniform_real_distribution<float> urdistr(arg1, arg2);
        for (int i = 0; i < rows * columns; i++)
        {
            data[i] = urdistr(gen);
        }
        break;
    }
    }
}

Matrix::Matrix(string path)
{
    ifstream file(path, std::ios::binary);

    if (!file.is_open())
    {
        cerr << "unable to open file" << endl;
    }

    file.read(reinterpret_cast<char *>(&this->rows), sizeof(int));
    file.read(reinterpret_cast<char *>(&this->columns), sizeof(int));
    file.read(reinterpret_cast<char *>(this->data.data()), sizeof(float) * this->rows * this->columns);

    if (data.size() != rows * columns)
    {
        cerr << "matrix dimensions does not match data" << endl;
    }

    file.close();
}

void Matrix::write(string path)
{
    ofstream file(path, std::ios::binary);

    if (!file)
    {
        cerr << "unable to open file" << endl;
    }

    file.write(reinterpret_cast<char *>(&this->rows), sizeof(int));
    file.write(reinterpret_cast<char *>(&this->columns), sizeof(int));
    file.write(reinterpret_cast<char *>(this->data.data()), sizeof(float) * this->rows * this->columns);

    file.close();
}

void Matrix::map(float (*function)(float))
{
    transform(data.begin(), data.end(), data.begin(), function);
}

void Matrix::resize(int rows, int columns)
{
    this->rows = rows;
    this->columns = columns;
    data.resize(rows * columns);
}

Matrix Matrix::operator+(const Matrix &other) const
{
    if (other.getColumns() != columns || other.getRows() != rows)
    {
        cerr << "matrix dimensions doesn't allow addition" << endl;
        return *this;
    }
    Matrix result(rows, columns);
#pragma omp parallel for collapse(2) schedule(dynamic)
    for (int r = 0; r < rows; r++)
    {
        for (int c = 0; c < columns; c++)
        {
            result[r][c] = other[r][c] + getData(r, c);
        }
    }
    return result;
}

Matrix Matrix::operator-(const Matrix &other) const
{
    if (other.getColumns() != columns || other.getRows() != rows)
    {
        cerr << "matrix dimensions doesn't allow subtraction" << endl;
        return *this;
    }
    Matrix result(rows, columns);
#pragma omp parallel for collapse(2) schedule(dynamic)
    for (int r = 0; r < rows; r++)
    {
        for (int c = 0; c < columns; c++)
        {
            result[r][c] = getData(r, c) - other[r][c];
        }
    }
    return result;
}

Matrix Matrix::operator*(const Matrix &other) const
{
    if (other.getRows() != columns)
    {
        cerr << "matrix dimensions doesn't allow multiplication" << endl;
        return *this;
    }
    Matrix result(rows, other.getColumns());
#pragma omp parallel for collapse(2) schedule(dynamic)
    for (int r = 0; r < rows; r++)
    {
        for (int c = 0; c < other.getColumns(); c++)
        {
            float sum = 0;
            for (int i = 0; i < columns; i++)
            {
                sum += getData(r, i) * other.getData(i, c);
            }
            result[r][c] = sum;
        }
    }
    return result;
}

Matrix &Matrix::operator=(const Matrix &other)
{
    this->resize(other.getRows(), other.getColumns());
    data = other.data;
    return *this;
}

Matrix &Matrix::operator+=(const Matrix &other)
{
    if (other.getColumns() != columns || other.getRows() != rows)
    {
        cerr << "matrix dimensions doesn't allow addition" << endl;
        return *this;
    }
#pragma omp parallel for collapse(2) schedule(dynamic)
    for (int r = 0; r < rows; r++)
    {
        for (int c = 0; c < columns; c++)
        {
            getData(r, c) += other[r][c];
        }
    }
    return *this;
}

Matrix &Matrix::operator-=(const Matrix &other)
{
    if (other.getColumns() != columns || other.getRows() != rows)
    {
        cerr << "matrix dimensions doesn't allow subtraction" << endl;
        return *this;
    }
#pragma omp parallel for collapse(2) schedule(dynamic)
    for (int r = 0; r < rows; r++)
    {
        for (int c = 0; c < columns; c++)
        {
            getData(r, c) -= other[r][c];
        }
    }
    return *this;
}

Matrix &Matrix::operator*=(const Matrix &other)
{
    if (other.getRows() != columns)
    {
        cerr << "matrix dimensions doesn't allow multiplication" << endl;
        return *this;
    }
    Matrix result(rows, other.getColumns());
#pragma omp parallel for collapse(2) schedule(dynamic)
    for (int r = 0; r < rows; r++)
    {
        for (int c = 0; c < other.getColumns(); c++)
        {
            float sum = 0;
            for (int i = 0; i < columns; i++)
            {
                sum += getData(r, i) * other.getData(i, c);
            }
            result[r][c] = sum;
        }
    }
    *this = result;
    return *this;
}
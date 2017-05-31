#ifndef MATH_H
#define MATH_H

#include <cmath>
#include <algorithm>

#define PI 3.1415926535897932384626433832795

#define Abs(x) std::abs(x)
#define Min(x, y) std::min(x, y)
#define Max(x, y) std::max(x, y)
#define Sqrt(x) std::sqrt(x)
#define Sin(rad) std::sin(rad)
#define Cos(rad) std::cos(rad)
#define Tan(rad) std::tan(rad)
#define SinCos(rad, si, co) sincosf(rad, si, co)
#define ACos(x) std::acos(x)
#define DegToRad(deg) ((deg) * 0.01745329251994329576923690768489)
#define Log2(x) std::log2(x)

template<class T> inline T Square(T x) { return x*x; }

// Vec2

struct Vec2
{
    float x, y;
};

inline Vec2 V2(float x, float y)
{
    Vec2 result;
    result.x = x;
    result.y = y;
    return result;
}

inline Vec2 V2(float s) { return V2(s, s); }

// Vec3 and Vec3d

#include "vec3.h"
#define vec3_t Vec3d
#define real_t double
#include "vec3.h"

inline Vec3 V3(float x, float y, float z)
{
    Vec3 result;
    result.x = x;
    result.y = y;
    result.z = z;
    return result;
}

inline Vec3 V3(float s) { return V3(s, s, s); }
inline Vec3 V3(const float *v) { return V3(v[0], v[1], v[2]); }
inline Vec3 V3(Vec3d v) { return V3(v.x, v.y, v.z); }

inline Vec3d V3d(double x, double y, double z)
{
    Vec3d result;
    result.x = x;
    result.y = y;
    result.z = z;
    return result;
}

inline Vec3d V3d(double s) { return V3d(s, s, s); }
inline Vec3d V3d(const float *v) { return V3d(v[0], v[1], v[2]); }

// Mat3

struct Mat3
{
    float data[3][3];

    inline float *operator[](int i) { return data[i]; }
    inline const float *operator[](int i) const { return data[i]; }
};

// operators

inline Mat3 operator*(const Mat3 &a, const Mat3 &b)
{
    Mat3 result;
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            result[i][j] =
                a[0][j] * b[i][0] +
                a[1][j] * b[i][1] +
                a[2][j] * b[i][2];
        }
    }
    return result;
}

// functions

inline Mat3 Mat3Identity()
{
    Mat3 result = {};
    result[0][0] = 1.0f;
    result[1][1] = 1.0f;
    result[2][2] = 1.0f;
    return result;
}

inline Mat3 Mat3RotationX(float rad)
{
    float si, co;
    SinCos(rad, &si, &co);
    Mat3 result = {};
    result[0][0] = 1.0f;
    result[1][1] = co;
    result[1][2] = si;
    result[2][1] = -si;
    result[2][2] = co;
    return result;
}

inline Mat3 Mat3RotationY(float rad)
{
    float si, co;
    SinCos(rad, &si, &co);
    Mat3 result = {};
    result[0][0] = co;
    result[0][2] = -si;
    result[1][1] = 1.0f;
    result[2][0] = si;
    result[2][2] = co;
    return result;
}

inline Mat3 Mat3RotationZ(float rad)
{
    float si, co;
    SinCos(rad, &si, &co);
    Mat3 result = {};
    result[0][0] = co;
    result[0][1] = si;
    result[1][0] = -si;
    result[1][1] = co;
    result[2][2] = 1.0f;
    return result;
}

inline Mat3 Mat3FromBasisVectors(Vec3 x, Vec3 y, Vec3 z)
{
    Mat3 result;
    result[0][0] = x.x; result[0][1] = x.y; result[0][2] = x.z;
    result[1][0] = y.x; result[1][1] = y.y; result[1][2] = y.z;
    result[2][0] = z.x; result[2][1] = z.y; result[2][2] = z.z;
    return result;
}

// Mat4

struct Mat4
{
    float data[4][4];

    float *operator[](int i) { return data[i]; }
    const float *operator[](int i) const { return data[i]; }
};

// operators

inline Mat4 operator*(const Mat4 &a, const Mat4 &b)
{
    Mat4 result;
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            result[i][j] =
                a[0][j] * b[i][0] +
                a[1][j] * b[i][1] +
                a[2][j] * b[i][2] +
                a[3][j] * b[i][3];
        }
    }
    return result;
}

// functions

inline Mat4 Mat4Identity()
{
    Mat4 result = {};
    result[0][0] = 1.0f;
    result[1][1] = 1.0f;
    result[2][2] = 1.0f;
    result[3][3] = 1.0f;
    return result;
}

inline void Mat4SetRotation(Mat4 &m, const Mat3 &R)
{
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; ++j)
        {
            m[i][j] = R[i][j];
        }
    }
}

inline void Mat4SetTranslation(Mat4 &m, Vec3 T)
{
    m[3][0] = T.x;
    m[3][1] = T.y;
    m[3][2] = T.z;
}

inline Mat4 Mat4ComposeTR(const Mat3 &R, Vec3 T)
{
    Mat4 result = {};
    Mat4SetRotation(result, R);
    Mat4SetTranslation(result, T);
    result[3][3] = 1.0f;
    return result;
}

inline Vec3 Mat4RotatePoint(const Mat4 &m, Vec3 v)
{
    Vec3 result;
    result.x = m[0][0] * v.x + m[1][0] * v.y + m[2][0] * v.z;
    result.y = m[0][1] * v.x + m[1][1] * v.y + m[2][1] * v.z;
    result.z = m[0][2] * v.x + m[1][2] * v.y + m[2][2] * v.z;
    return result;
}

inline Mat4 Mat4InverseTR(const Mat4 &m)
{
    Mat4 result = {};
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            result[i][j] = m[j][i];
        }
    }

    Vec3 T = Mat4RotatePoint(result, V3(m[3]));
    Mat4SetTranslation(result, -T);
    result[3][3] = 1.0f;
    return result;
}

// projections

inline Mat4 Mat4PerspectiveLH(float fovy_rad, float aspect_ratio,
                              float z_near, float z_far)
{
    Mat4 result = {};
    result[1][1] = 1.0f / Tan(fovy_rad * 0.5f);
    result[0][0] = result[1][1] / aspect_ratio;
    result[2][2] = (z_far + z_near) / (z_far - z_near);
    result[2][3] = 1.0f;
    result[3][2] = 2.0f * z_far * z_near / (z_near - z_far);
    return result;
}

/*
 * Maps zNear to -1 and zFar to 1.
 * In right handed, zNear would map to 1 and zFar to -1.
 */
inline Mat4 Mat4OrthoLH(float left, float right, float bottom, float top,
                        float z_near, float z_far)
{
    Mat4 result = {};
    result[0][0] = 2.0f / (right - left);
    result[1][1] = 2.0f / (top - bottom);
    result[2][2] = 2.0f / (z_far - z_near);
    result[3][0] = (right + left) / (left - right);
    result[3][1] = (top + bottom) / (bottom - top);
    result[3][2] = (z_far + z_near) / (z_near - z_far);
    result[3][3] = 1.0f;
    return result;
}

#endif // MATH_H

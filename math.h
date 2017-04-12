#ifndef MATH_H
#define MATH_H

#include <algorithm>

// Math types
//

template<class T>
struct Vec3
{
    T x, y, z;
};

typedef Vec3<float> Vec3f;
typedef Vec3<double> Vec3d;


// Functions
//

template<class T>
inline T Abs(T s)
{
    return std::abs(s);
}

template<class T>
inline T Min(T a, T b)
{
    return std::min(a, b);
}

template<class T>
inline T Max(T a, T b)
{
    return std::max(a, b);
}

template<class T>
inline T Square(T s)
{
    return s*s;
}

template<class T>
inline T Sqrt(T s)
{
    return std::sqrt(s);
}

// Vec3

template<class T>
inline Vec3<T> V3(T x, T y, T z)
{
    Vec3<T> result;
    result.x = x;
    result.y = y;
    result.z = z;
    return result;
}

template<class T>
inline Vec3<T> V3(T s)
{
    return V3(s, s, s);
}

template<class T>
inline Vec3<T> operator+(Vec3<T> a, Vec3<T> b)
{
    Vec3<T> result;
    result.x = a.x + b.x;
    result.y = a.y + b.y;
    result.z = a.z + b.z;
    return result;
}

template<class T>
inline Vec3<T> operator*(Vec3<T> a, T b)
{
    Vec3<T> result;
    result.x = a.x * b;
    result.y = a.y * b;
    result.z = a.z * b;
    return result;
}

template<class T>
inline Vec3<T> operator*(T a, Vec3<T> b)
{
    return b*a;
}

template<class T>
inline Vec3<T> operator/(Vec3<T> a, T b)
{
    return a*(1.0/b);
}

template<class T>
inline T Dot(Vec3<T> a, Vec3<T> b)
{
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

template<class T>
inline T LengthSq(Vec3<T> v)
{
    return Dot(v, v);
}

template<class T>
inline T Length(Vec3<T> v)
{
    return Sqrt(LengthSq(v));
}

template<class T>
inline Vec3<T> Normalize(Vec3<T> v)
{
    return v / Length(v);
}

template<class T>
inline Vec3<T> SafeNormalize(Vec3<T> v, T epsilon = 0.0001)
{
    T len2 = LengthSq(v);
    if (len2 < epsilon)
        return V3(0.0);
    return v / Sqrt(len2);
}

#endif // MATH_H

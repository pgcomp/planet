/* Vector 3

We don't want templates.
Define "vec3_t" and "real_t" before
including this file to a _guarded_ header.

Example:
// Use defaults: vec3_t = Vec3, realt_t = float
#include "vec3.h"
#define vec3_t Vec3d
#define real_t double
#include "vec3.h"
// Vec3 and Vec3d are now available

*/

#if !defined(vec3_t)
#define vec3_t Vec3
#endif

#if !defined(real_t)
#define real_t float
#endif

struct vec3_t
{
    real_t x, y, z;

    inline vec3_t &operator+=(vec3_t v) { x+=v.x; y+=v.y; z+=v.z; return *this; }
    inline vec3_t &operator-=(vec3_t v) { x-=v.x; y-=v.y; z-=v.z; return *this; }
    inline vec3_t &operator*=(real_t s) { x *= s; y *= s; z *= s; return *this; }
    inline vec3_t &operator/=(real_t s) { x /= s; y /= s; z /= s; return *this; }
};

// operators

inline vec3_t operator+(vec3_t a, vec3_t b) { return a += b; }
inline vec3_t operator-(vec3_t a, vec3_t b) { return a -= b; }
inline vec3_t operator*(vec3_t a, real_t b) { return a *= b; }
inline vec3_t operator*(real_t a, vec3_t b) { return b *= a; }
inline vec3_t operator/(vec3_t a, real_t b) { return a /= b; }
inline vec3_t operator-(vec3_t v) { return v *= real_t(-1.0); }

// functions

inline real_t Dot(vec3_t a, vec3_t b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
inline real_t LengthSq(vec3_t v)    { return Dot(v, v); }
inline real_t Length(vec3_t v)      { return Sqrt(LengthSq(v)); }
inline vec3_t Normalize(vec3_t v)   { return v / Length(v); }

inline vec3_t SafeNormalize(vec3_t v, real_t epsilon = 0.0001)
{
    real_t len2 = LengthSq(v);
    if (len2 < epsilon)
        return (vec3_t){};
    return v / Sqrt(len2);
}

#undef vec3_t
#undef real_t

#ifndef PERLIN_H
#define PERLIN_H

float PerlinNoise3(double x, double y, double z);

#endif

#ifdef PERLIN_IMPLEMENTATION

static unsigned char perlin_random_table[256] =
{
    211,222, 90, 42,136, 37,204,126, 22,101,213,137,251, 28,247,205,
    185,176,200,206,243,130,252,188, 19,235,231,  1,170,109, 11, 31,
     58,134,230,148, 65,184,250,226,129,197,135, 99,201,  5, 40,220,
    132,218, 15,110,120,239,151, 35,141, 70,217,  7,107,150,178,162,
    160, 93,164,118,174, 29, 45, 84,207, 81,  8, 64, 43,244,203, 67,
     95, 25, 69,  3,183,242, 94,172,121,144,122,249, 61,159,240, 59,
    193,157,224, 52, 71,112, 32,167,155,165,177,255, 78, 10, 26,149,
    124,133,140,189,233, 60, 96,254, 50,236,131,215, 49, 79, 54,214,
    196,104,234, 18,181, 53,152,116,127, 30,182,  6, 98,146,208,102,
    221,241, 48,228, 73, 82,245,142,105, 80, 34,246, 23,139,238, 97,
     51,190,186,232, 44, 91, 87,173, 16,168, 46, 75,199,138,198, 33,
     24, 66,225,195,169,100, 88,237, 38, 57,  0,  4, 86, 14,253,115,
     47,212,180,171,163, 63,194,227,210, 62, 12, 89,161,192, 39,166,
    128,123, 17,223,106,117,229,108, 76,145,125,219,175, 36,202,114,
    153, 72,209, 27, 83, 85, 13, 68,147,158,187,179,156,154, 56, 77,
     20,143,119,103,113,191,  9, 41, 74,216,  2,111, 21, 92,248, 55,
};

static float perlin_vectors[12][3] =
{
    { -1.0f, -1.0f,  0.0f },
    { -1.0f,  1.0f,  0.0f },
    {  1.0f, -1.0f,  0.0f },
    {  1.0f,  1.0f,  0.0f },
    { -1.0f,  0.0f, -1.0f },
    { -1.0f,  0.0f,  1.0f },
    {  1.0f,  0.0f, -1.0f },
    {  1.0f,  0.0f,  1.0f },
    {  0.0f, -1.0f, -1.0f },
    {  0.0f, -1.0f,  1.0f },
    {  0.0f,  1.0f, -1.0f },
    {  0.0f,  1.0f,  1.0f },
};

inline int PerlinRandom(int seed)
{
    return perlin_random_table[seed & 255];
}

float PerlinGradient(float x, float y, float z, int ix, int iy, int iz)
{
    int rand = PerlinRandom(PerlinRandom(PerlinRandom(ix) + iy) + iz);
    float *vec = perlin_vectors[rand % 12];
    return x*vec[0] + y*vec[1] + z*vec[2];
}

float PerlinNoise3(double x, double y, double z)
{
#define FLOOR(x) (int)(((x) < 0.0) ? ((x) - 1.0) : (x))
    int ix = FLOOR(x);
    int iy = FLOOR(y);
    int iz = FLOOR(z);
#undef FLOOR

    x -= ix;
    y -= iy;
    z -= iz;

#define CURVE(x) ((((x)*6.0f - 15.0f)*(x) + 10.0f)*(x)*(x)*(x))
    float u = CURVE(x);
    float v = CURVE(y);
    float w = CURVE(z);
#undef CURVE

    float grad0 = PerlinGradient(x,     y,     z,     ix,     iy,     iz);
    float grad1 = PerlinGradient(x - 1, y,     z,     ix + 1, iy,     iz);
    float grad2 = PerlinGradient(x,     y - 1, z,     ix,     iy + 1, iz);
    float grad3 = PerlinGradient(x - 1, y - 1, z,     ix + 1, iy + 1, iz);
    float grad4 = PerlinGradient(x,     y,     z - 1, ix,     iy,     iz + 1);
    float grad5 = PerlinGradient(x - 1, y,     z - 1, ix + 1, iy,     iz + 1);
    float grad6 = PerlinGradient(x,     y - 1, z - 1, ix,     iy + 1, iz + 1);
    float grad7 = PerlinGradient(x - 1, y - 1, z - 1, ix + 1, iy + 1, iz + 1);

#define LERP(a, b, t) ((a) + ((b) - (a)) * (t))
    float lerp0 = LERP(grad0, grad1, u);
    float lerp1 = LERP(grad2, grad3, u);
    float lerp2 = LERP(grad4, grad5, u);
    float lerp3 = LERP(grad6, grad7, u);

    float lerp4 = LERP(lerp0, lerp1, v);
    float lerp5 = LERP(lerp2, lerp3, v);

    return LERP(lerp4, lerp5, w);
#undef LERP
}

#endif

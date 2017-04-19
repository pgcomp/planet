#ifndef RENDER_H
#define RENDER_H

#include <GL/glew.h>
#include "math.h"

#define GLSL(vs, fs) \
"#if defined(VERTEX_SHADER)\n" #vs "\n#else\n" #fs "\n#endif\n"


#define MAX_VERTEX_ATTRIBUTES 16

struct VertexFormat
{
    struct Attribute
    {
        int size;
        GLenum type;
        bool normalize;
        void *pointer;
    };

    int vertex_size;
    Attribute attributes[MAX_VERTEX_ATTRIBUTES];
};

struct Texture
{
    GLuint texture;
    int bind_index;
};

struct Uniform
{
    enum Type
    {
        FLOAT = 1,
        VEC2  = 2,
        VEC3  = 3,
        MAT4  = 16
    };

    GLint loc;
    Type type;
    int count;
    union
    {
        float f[16];
        Vec2  v2[8];
        Vec3  v3[5];
        //Vec4  v4[4];
        Mat4  m4;
    } value;
};

struct TexUniforms
{
    Uniform corners;
    Uniform pixel_size;
};

struct DrawItem
{
    GLuint shader;
    GLuint vertex_array;
    Texture *textures;
    Uniform *uniforms;
    int texture_count;
    int uniform_count;
    GLenum primitive_mode;
    GLenum index_type;
    int first;
    int count;
    bool draw_arrays;
};


// Geometry

GLuint CreateVertexBuffer(unsigned int size, const void *data);
GLuint CreateIndexBuffer(unsigned int size, const void *data);
void AddVertexAttribute(VertexFormat &format, int index, int size,
                        GLenum type, bool normalize);
GLuint CreateVertexArray(GLuint vertex_buffer, VertexFormat &format,
                         GLuint index_buffer);

// Shaders

GLuint CreateShaderFromSource(const char *source);

inline int GetUniformCountFromSize(unsigned int size)
{
    return size / sizeof(Uniform);
}

void InitUniform(Uniform &u, GLuint shader, const char *name,
                 Uniform::Type type, int count = 1);
void InitTexUniforms(TexUniforms &us, GLuint shader, const char *sampler_name);

// Textures

GLuint CreateTexture2D(unsigned int w, unsigned int h, GLenum fmt,
                       GLenum type, const void *data);
void DeleteTexture(GLuint texture);

// Draw

void Draw(DrawItem &d);

#endif // RENDER_H

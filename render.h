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


struct Uniform
{
    enum Type
    {
        LIST_END, FLOAT, VEC3, MAT4
    };

    GLint loc;
    Type type;
    union
    {
        float f;
        Vec3  v3;
        Mat4  m4;
        float data[1];
    } value;
};

struct DrawItem
{
    GLuint shader;
    GLuint vertex_array;
    Uniform *uniforms;
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

void InitUniform(Uniform &u, GLuint shader, const char *name, float value);
void InitUniform(Uniform &u, GLuint shader, const char *name, Vec3 value);
void InitUniform(Uniform &u, GLuint shader, const char *name, const Mat4 &value);

// Draw

void Draw(DrawItem &d);

#endif // RENDER_H

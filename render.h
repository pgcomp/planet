#ifndef RENDER_H
#define RENDER_H

#include <GL/glew.h>

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


struct DrawItem
{
    GLuint shader;
    GLuint vertex_array;
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

// Draw

void Draw(DrawItem &d);

#endif // RENDER_H

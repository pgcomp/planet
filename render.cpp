#include "render.h"
#include "logging.h"
#include "pp.h"
#include <cstring>
#include <cassert>

// Geometry

inline GLuint CreateBuffer(GLenum target, unsigned int size, const void *data)
{
    glBindVertexArray(0);

    GLuint buffer;
    glGenBuffers(1, &buffer);
    glBindBuffer(target, buffer);
    glBufferData(target, size, data, GL_STATIC_DRAW);
    return buffer;
}

GLuint CreateVertexBuffer(unsigned int size, const void *data)
{
    return CreateBuffer(GL_ARRAY_BUFFER, size, data);
}

GLuint CreateIndexBuffer(unsigned int size, const void *data)
{
    return CreateBuffer(GL_ELEMENT_ARRAY_BUFFER, size, data);
}

inline int GetAttributeTypeSize(GLenum type)
{
    switch (type)
    {
        case GL_FLOAT: return sizeof(float);
        default: assert(0 && "Invalid attribute type!");
    }
    return 0;
}

void AddVertexAttribute(VertexFormat &format, int index, int size,
                        GLenum type, bool normalize)
{
    assert(0 <= index && index < MAX_VERTEX_ATTRIBUTES);
    assert(1 <= size && size <= 4);
    assert(format.attributes[index].size == 0);
    format.attributes[index].size = size;
    format.attributes[index].type = type;
    format.attributes[index].normalize = normalize;
    format.attributes[index].pointer = ((char*)0 + format.vertex_size);
    format.vertex_size += size * GetAttributeTypeSize(type);
}

GLuint CreateVertexArray(GLuint vertex_buffer, VertexFormat &format,
                         GLuint index_buffer)
{
    assert(vertex_buffer != 0);
    assert(format.vertex_size > 0);

    GLuint vertex_array;
    glGenVertexArrays(1, &vertex_array);
    glBindVertexArray(vertex_array);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer);

    for (int index = 0; index < MAX_VERTEX_ATTRIBUTES; index++)
    {
        auto &a = format.attributes[index];
        if (a.size == 0) continue;

        glEnableVertexAttribArray(index);
        glVertexAttribPointer(index, a.size, a.type,
                              a.normalize ? GL_TRUE : GL_FALSE,
                              format.vertex_size, a.pointer);
    }

    return vertex_array;
}

// Shaders

GLuint CompileShader(GLenum type, const char *source)
{
    const char *source_list[3];
    source_list[0] = "#version 140\n";
    source_list[1] = "\n";
    source_list[2] = source;
    if (type == GL_VERTEX_SHADER)
    {
        source_list[1] =
            "#define VERTEX_SHADER\n"
            "#define ATTRIBUTE(type, name, index) in type name\n";
    }

    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 3, source_list, 0);
    glCompileShader(shader);

    GLint status;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
    if (status == GL_TRUE)
        return shader;

    char info[256];
    glGetShaderInfoLog(shader, sizeof(info), 0, info);
    LOG_ERROR("Couldn't compile shader:\n%s\n", info);
    glDeleteShader(shader);
    return 0;
}

// Parse utils
inline bool IsSpace(char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n'; }
inline bool IsAlpha(char c) { return ('a' <= c && c <= 'z') || ('A' <= c && c <= 'Z'); }
inline bool IsNumber(char c) { return '0' <=c && c <= '9'; }
inline bool IsAlphaNum(char c) { return IsAlpha(c) || IsNumber(c); }
inline bool IsIdent(char c) { return IsAlphaNum(c) || c == '_'; }
inline const char *SkipSpace(const char *s) { while (*s && IsSpace(*s)) s++; return s; }

GLuint CreateShaderFromSource(const char *source)
{
    GLuint vs = CompileShader(GL_VERTEX_SHADER, source);
    GLuint fs = CompileShader(GL_FRAGMENT_SHADER, source);
    DEFER(if (vs != 0) { glDeleteShader(vs); }, GLuint, vs);
    DEFER(if (fs != 0) { glDeleteShader(fs); }, GLuint, fs);

    if (vs == 0 || fs == 0)
        return 0;

    GLuint p = glCreateProgram();

    // Parse attributes
    // NOTE: We can be sloppy because the source has passed the compilation
    char prev = '\0';
    for (const char *c = source; *c; c++)
    {
        if (*c == 'A' && !IsIdent(prev))
        {
            const char *ATTRIBUTE = "ATTRIBUTE";
            while (*ATTRIBUTE && *c == *ATTRIBUTE)
            {
                ATTRIBUTE++;
                c++;
            }

            if (*ATTRIBUTE == '\0')
            {
                c = SkipSpace(c);
                if (*c == '(')
                {
                    // Sloppy: not checking eof
                    // skip attribute type
                    while (*c++ != ','); // skips the comma automatically
                    // parse attribute name
                    c = SkipSpace(c);
                    const char *name = c;
                    while (IsIdent(*c)) c++;
                    int name_len = c - name;
                    while (*c++ != ','); // skips the comma automatically
                    // parse attribute index
                    c = SkipSpace(c);
                    int index = 0;
                    bool is_num = IsNumber(*c);
                    while (IsNumber(*c))
                    {
                        index *= 10;
                        index += *c - '0';
                        c++;
                    }
                    while (*c++ != ')'); // skips the paren automatically

                    do {
                        char buf[64];
                        if (0 < name_len && name_len < (int)sizeof(buf))
                        {
                            memcpy(buf, name, name_len);
                            buf[name_len] = '\0';
                        }
                        else
                        {
                            LOG_ERROR("Vertex attribute name too long.");
                            break;
                        }
                        if (!is_num || index >= MAX_VERTEX_ATTRIBUTES)
                        {
                            LOG_ERROR("Expected attribute index between 0-%d.",
                                      MAX_VERTEX_ATTRIBUTES);
                            break;
                        }
                        glBindAttribLocation(p, index, buf);
                    } while(0);
                }
            }

            c--;
        }

        prev = *c;
    }

    glAttachShader(p, vs);
    glAttachShader(p, fs);
    glLinkProgram(p);
    glDetachShader(p, vs);
    glDetachShader(p, fs);

    GLint status;
    glGetProgramiv(p, GL_LINK_STATUS, &status);
    if (status == GL_TRUE)
        return {p};

    char info[256];
    glGetProgramInfoLog(p, 256, 0, info);
    LOG_ERROR("CreateShader() failed:\n%s\n", info);
    glDeleteProgram(p);
    return {};
}

void InitUniform(Uniform &u, GLuint shader, const char *name, float value)
{
    u.loc = glGetUniformLocation(shader, name);
    u.type = Uniform::FLOAT;
    u.value.f = value;
}

void InitUniform(Uniform &u, GLuint shader, const char *name, Vec3 value)
{
    u.loc = glGetUniformLocation(shader, name);
    u.type = Uniform::VEC3;
    u.value.v3 = value;
}

void InitUniform(Uniform &u, GLuint shader, const char *name, const Mat4 &value)
{
    u.loc = glGetUniformLocation(shader, name);
    u.type = Uniform::MAT4;
    u.value.m4 = value;
}

// Draw

inline int GetIndexTypeSize(GLenum type)
{
    switch (type)
    {
        case GL_UNSIGNED_INT: return sizeof(uint32_t);
        default: assert(0 && "Invalid index type!");
    }
    return 0;
}

void Draw(DrawItem &d)
{
    glUseProgram(d.shader);
    glBindVertexArray(d.vertex_array);

    for (int i = 0; i < d.uniform_count; i++)
    {
        Uniform &u = d.uniforms[i];
        switch (u.type)
        {
            case Uniform::FLOAT: glUniform1fv(u.loc, 1, u.value.data); break;
            case Uniform::VEC3:  glUniform3fv(u.loc, 3, u.value.data); break;
            case Uniform::MAT4:
            glUniformMatrix4fv(u.loc, 1, GL_FALSE, u.value.data); break;
            default: assert(0 && "Invalid uniform type!");
        }
    }

    if (d.draw_arrays)
    {
        glDrawArrays(d.primitive_mode, d.first, d.count);
    }
    else
    {
        void *offset = (char*)0 + d.first * GetIndexTypeSize(d.index_type);
        glDrawElements(GL_TRIANGLES, d.count, d.index_type, offset);
    }
}

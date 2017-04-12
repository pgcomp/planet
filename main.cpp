#include <SDL2/SDL.h>
#include "render.h"
#include "math.h"
#include "logging.h"
#include "pp.h"

#define ArrayCount(arr) (sizeof(arr)/sizeof(*arr))


int main(int argc, char **argv)
{
    // Initialize SDL
    //
    if (SDL_Init(SDL_INIT_VIDEO) < 0)
    {
        LOG_ERROR("Couldn't initialize SDL: %s", SDL_GetError());
        return 1;
    }

    DEFER(SDL_Quit());


    // Create window
    //
    SDL_GL_SetAttribute(SDL_GL_RED_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 1);

    SDL_Window *window = SDL_CreateWindow("Test",
                                          SDL_WINDOWPOS_UNDEFINED,
                                          SDL_WINDOWPOS_UNDEFINED,
                                          800, 600,
                                          SDL_WINDOW_OPENGL);

    if (window == nullptr)
    {
        LOG_ERROR("Couldn't create window: %s", SDL_GetError());
        return 1;
    }

    DEFER(SDL_DestroyWindow(window), SDL_Window*, window);


    // Create GL context
    //
    SDL_GLContext context = SDL_GL_CreateContext(window);

    if (context == nullptr)
    {
        LOG_ERROR("Couldn't create GL context: %s", SDL_GetError());
        return 1;
    }

    DEFER(SDL_GL_DeleteContext(context), SDL_GLContext, context);


    // Initialize GLEW
    //
    glewExperimental = GL_TRUE;

    GLenum error = glewInit();

    if (error != GLEW_OK)
    {
        LOG_ERROR("Couldn't initialize GLEW: %s", glewGetErrorString(error));
        return 1;
    }

    if (!GLEW_VERSION_3_1)
    {
        LOG_ERROR("Didn't get GL version 3.1.");
        return 1;
    }

    // Setup things

    glDepthFunc(GL_LEQUAL);
    glEnable(GL_DEPTH_TEST);

    glFrontFace(GL_CCW);
    glCullFace(GL_BACK);
    glEnable(GL_CULL_FACE);

    const char *shader_source =
        GLSL(ATTRIBUTE(vec3, Position, 0);
             void main() {
             gl_Position = vec4(Position.xy, 0.0, 1.0);
             },
             void main() {
             gl_FragColor = vec4(1.0, 0.0, 1.0, 1.0);
             });

    GLuint shader = CreateShaderFromSource(shader_source);

    if (shader == 0)
    {
        return 1;
    }

    DrawItem cube = {};
    {
        //Vec3f o = V3(2.0f, -1.5f, 5.0f);
        Vec3f o = V3(0.0f);
        float k = 1.0f;
        Vec3f verts[] =
        {
            o + V3(-k, -k, -k), // 0
            o + V3( k, -k, -k), // 1
            o + V3( k,  k, -k), // 2
            o + V3(-k,  k, -k), // 3

            o + V3(-k, -k,  k), // 4
            o + V3( k, -k,  k), // 5
            o + V3( k,  k,  k), // 6
            o + V3(-k,  k,  k), // 7
        };

#define QUAD(a, b, c, d) a, b, c, a, c, d
        unsigned int indices[] =
        {
            QUAD(0, 1, 2, 3), // front
            QUAD(1, 5, 6, 2), // right
            QUAD(5, 4, 7, 6), // back
            QUAD(4, 0, 3, 7), // left
            QUAD(3, 2, 6, 7), // top
            QUAD(4, 5, 1, 0), // bottom
        };
#undef QUAD

        VertexFormat fmt = {};
        AddVertexAttribute(fmt, 0, 3, GL_FLOAT, false);

        GLuint vertex_buffer = CreateVertexBuffer(sizeof(verts), verts);
        GLuint index_buffer = CreateIndexBuffer(sizeof(indices), indices);
        GLuint vertex_array = CreateVertexArray(vertex_buffer, fmt, index_buffer);

        cube.shader = shader;
        cube.vertex_array = vertex_array;
        cube.primitive_mode = GL_TRIANGLES;
        cube.index_type = GL_UNSIGNED_INT;
        cube.first = 0;
        cube.count = ArrayCount(indices);
        cube.draw_arrays = false;
    }

    // Start looping
    //
    bool running = true;

    while (running)
    {
        // Handle events
        for (SDL_Event event; SDL_PollEvent(&event); )
        {
            if (event.type == SDL_QUIT)
            {
                running = false;
                break;
            }

            if (event.type == SDL_KEYDOWN)
            {
                if (event.key.keysym.scancode == SDL_SCANCODE_ESCAPE)
                {
                    running = false;
                    break;
                }
            }
        }

        // Render
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);

        Draw(cube);

        SDL_GL_SwapWindow(window);
        SDL_Delay(10);

        // Check GL errors
        for (GLenum err; (err = glGetError()) != GL_NO_ERROR; )
        {
            switch (err)
            {
#define CASE(e) case e: LOG_ERROR("GL error: " #e); break
                CASE(GL_INVALID_ENUM);
                CASE(GL_INVALID_VALUE);
                CASE(GL_INVALID_OPERATION);
                CASE(GL_INVALID_FRAMEBUFFER_OPERATION);
                CASE(GL_OUT_OF_MEMORY);
                CASE(GL_STACK_UNDERFLOW);
                CASE(GL_STACK_OVERFLOW);
                default: LOG_ERROR("Unknown GL error: %d", err);
#undef CASE
            }
        }
    }

    // Done
    return 0;
}

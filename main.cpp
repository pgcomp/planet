#include <SDL2/SDL.h>
#include <GL/glew.h>
#include <cstdio>

#define LOG_ERROR(fmt, ...) fprintf(stderr, "[ERROR] " fmt "\n", ##__VA_ARGS__)

#define JOIN__(a, b) a##b
#define JOIN_(a, b) JOIN__(a, b)
#define JOIN(a, b) JOIN_(a, b)

#define NUM_VA_ARGS_(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, N, ...) N
#define NUM_VA_ARGS(...) NUM_VA_ARGS_(__VA_ARGS__, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1)

#define DEFER1(code, line) struct _D##line { ~_D##line() { code; } } _d##line
#define DEFER3(code, line, T, a) struct _D##line { T a; _D##line(T a_):a(a_){} ~_D##line() { code; } } _d##line(a)
#define DEFER_(code, line, ...) JOIN(DEFER, NUM_VA_ARGS(1, ##__VA_ARGS__))(code, line, ##__VA_ARGS__)
//#define DEFER(code, ...) DEFER_(printf(#code"\n"); code, __LINE__, ##__VA_ARGS__)
#define DEFER(code, ...) DEFER_(code, __LINE__, ##__VA_ARGS__)



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

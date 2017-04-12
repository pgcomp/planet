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

    int window_w = 800;
    int window_h = 600;

    SDL_Window *window = SDL_CreateWindow("Test",
                                          SDL_WINDOWPOS_UNDEFINED,
                                          SDL_WINDOWPOS_UNDEFINED,
                                          window_w, window_h,
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

    glPolygonMode(GL_FRONT, GL_LINE);

    Mat4 proj;
    {
        float fovy = DegToRad(50.0f);
        float aspect = (float)window_w/window_h;
        proj = Mat4PerspectiveLH(fovy, aspect, 0.1f, 100.0f);
    }

    struct
    {
        Vec3 position;
        Vec3 angles;
    } cam = {};

    const char *shader_source =
        GLSL(ATTRIBUTE(vec3, Position, 0);
             uniform mat4 Projection;
             uniform mat4 View;
             void main() {
             gl_Position = Projection * View * vec4(Position, 1.0);
             },
             void main() {
             gl_FragColor = vec4(1.0, 0.0, 1.0, 1.0);
             });

    GLuint shader = CreateShaderFromSource(shader_source);

    if (shader == 0)
    {
        return 1;
    }

    Uniform uniforms[2];

    DrawItem cube = {};
    {
        Vec3 o = V3(2.0f, -1.5f, 10.0f);
        //Vec3 o = V3(0.0f);
        float k = 1.0f;
        Vec3 verts[] =
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

        InitUniform(uniforms[0], shader, "Projection", proj);
        InitUniform(uniforms[1], shader, "View", Mat4Identity());

        cube.shader = shader;
        cube.vertex_array = vertex_array;
        cube.uniform_count = ArrayCount(uniforms);
        cube.uniforms = uniforms;
        cube.primitive_mode = GL_TRIANGLES;
        cube.index_type = GL_UNSIGNED_INT;
        cube.first = 0;
        cube.count = ArrayCount(indices);
        cube.draw_arrays = false;
    }

    // Start looping
    //
    Uint32 last_t = SDL_GetTicks();
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

        const Uint8 *keys = SDL_GetKeyboardState(nullptr);

        float dt;
        {
            Uint32 ticks = SDL_GetTicks();
            Uint32 delta = ticks - last_t;
            last_t = ticks;
            dt = delta / 1000.0;
        }

        float move_speed = 3.0f; // m/s
        float look_speed = 2.0f; // rad/s

        Vec3 move = V3(keys[SDL_SCANCODE_D] - keys[SDL_SCANCODE_A], 0,
                       keys[SDL_SCANCODE_W] - keys[SDL_SCANCODE_S]);
        Vec3 look = V3(keys[SDL_SCANCODE_DOWN] - keys[SDL_SCANCODE_UP],
                       keys[SDL_SCANCODE_RIGHT] - keys[SDL_SCANCODE_LEFT], 0);

        cam.angles += look * look_speed * dt;

        Mat3 rotation = (Mat3RotationY(cam.angles.y) *
                         Mat3RotationX(cam.angles.x) *
                         Mat3RotationZ(cam.angles.z));

        cam.position += (V3(rotation[0]) * move.x +
                         V3(rotation[1]) * move.y +
                         V3(rotation[2]) * move.z) * move_speed * dt;

        Mat4 view = Mat4ComposeTR(rotation, cam.position);
        uniforms[1].value.m4 = Mat4InverseTR(view);

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

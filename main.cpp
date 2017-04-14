#include <SDL2/SDL.h>
#include "render.h"
#include "list.h"
#include "logging.h"
#include "pp.h"

#define ArrayCount(arr) (sizeof(arr)/sizeof(*arr))

// Not used anywhere
inline Vec3 Slerp(Vec3 a, Vec3 b, float t)
{
    float theta = ACos(Dot(a, b) / (Length(a)*Length(b)));
    return (Sin((1.0f - t)*theta)*a + Sin(t*theta)*b) / Sin(theta);
}

struct QVert
{
    Vec3d p, n;
};

struct Quad
{
    QVert verts[4];
};

struct Planet
{
    double radius;
    DrawItem patch;
    struct
    {
        Uniform proj;
        Uniform view;
        Uniform p[4];
        Uniform n[4];
    } uniforms;
    List<Quad> quads;
};

bool InitPlanet(Planet &p, double radius)
{
    const char *shader_source =
        GLSL(ATTRIBUTE(vec2, UV, 0);

             uniform mat4 Projection;
             uniform mat4 View;
             uniform vec3 P[4];
             uniform vec3 N[4];

             struct V { vec3 p; vec3 n; };

             V interpolate(V v0, V v1, float t) {
             // slerp normal
             float theta2 = acos(dot(v0.n, v1.n));
             float k = 1.0 - t;
             vec3 n = normalize(sin(k*theta2)*v0.n + sin(t*theta2)*v1.n);

             // compute position
             float theta = theta2 * 0.5;
             float gamma = theta - theta2*t;
             float tan_theta = tan(theta);
             float x = 1.0 - tan(gamma)/tan_theta;
             float y = 1.0/sin(theta) - 1.0/(cos(gamma)*tan_theta);
             vec3 v = (v1.p - v0.p) * 0.5;
             vec3 p = v0.p + x*v + y*n*length(v);

             V result;
             result.p = p;
             result.n = n;
             return result;
             }

             void main() {
             V a; a.p = P[0]; a.n = N[0];
             V b; b.p = P[1]; b.n = N[1];
             V c; c.p = P[2]; c.n = N[2];
             V d; d.p = P[3]; d.n = N[3];
             V p = interpolate(a, b, UV.x);
             V q = interpolate(c, d, UV.x);
             V v = interpolate(p, q, UV.y);
             gl_Position = Projection * View * vec4(v.p, 1.0);
             },

             void main() {
             gl_FragColor = vec4(0.0, 1.0, 1.0, 1.0);
             });

    GLuint shader = CreateShaderFromSource(shader_source);
    if (shader == 0)
    {
        return false;
    }


    struct Vert
    {
        Vec2 uv;
    };

    const int patch_size_in_verts = 30; // configurable
    const int patch_size_in_quads = patch_size_in_verts - 1;
    const int verts_total = Square(patch_size_in_verts);
    const int indices_per_reset = 2;
    const int indices_per_strip =
        2 + patch_size_in_quads*2 + indices_per_reset;
    const int indices_total =
        patch_size_in_quads*indices_per_strip - indices_per_reset;

    Vert verts[verts_total];
    {
        int n = 0;
        for (int y = 0; y < patch_size_in_verts; ++y)
        {
            for (int x = 0; x < patch_size_in_verts; ++x)
            {
                Vert &v = verts[n++];
                v.uv.x = (double)x / patch_size_in_quads;
                v.uv.y = (double)y / patch_size_in_quads;
            }
        }
        assert(n == verts_total);
    }

    unsigned int indices[indices_total];
    {
        int n = 0;
        unsigned int v0 = patch_size_in_verts;
        unsigned int v1 = 0;
        for (int y = 0; y < patch_size_in_quads; ++y)
        {
            for (int x = 0; x < patch_size_in_verts; ++x)
            {
                indices[n++] = v0++;
                indices[n++] = v1++;
            }
            if (y+1 < patch_size_in_quads)
            {
                // reset
                indices[n++] = v1-1;
                indices[n++] = v0;
            }
        }
        assert(n == indices_total);
    }

    VertexFormat vf = {};
    AddVertexAttribute(vf, 0, 2, GL_FLOAT, false);

    GLuint vertex_buffer = CreateVertexBuffer(sizeof(verts), verts);
    GLuint index_buffer = CreateIndexBuffer(sizeof(indices), indices);
    GLuint vertex_array = CreateVertexArray(vertex_buffer, vf, index_buffer);

    InitUniform(p.uniforms.proj, shader, "Projection", Mat4Identity());
    InitUniform(p.uniforms.view, shader, "View", Mat4Identity());
    for (int i = 0; i < 4; ++i)
    {
        char P[] = "P[_]"; P[2] = '0' + i;
        char N[] = "N[_]"; N[2] = '0' + i;
        InitUniform(p.uniforms.p[i], shader, P, V3(0.0f));
        InitUniform(p.uniforms.n[i], shader, N, V3(0.0f));
    };

    p.radius = radius;

    DrawItem &d = p.patch;
    d.shader = shader;
    d.vertex_array = vertex_array;
    d.uniforms = &p.uniforms.proj;
    d.uniform_count = GetUniformCountFromSize(sizeof(p.uniforms));
    d.primitive_mode = GL_TRIANGLE_STRIP;
    d.index_type = GL_UNSIGNED_INT;
    d.first = 0;
    d.count = ArrayCount(indices);
    d.draw_arrays = false;

    return true;
}

void ProcessQuad(Planet &p, const Quad &q, int depth)
{
    if (depth == 3)
    {
        ListAdd(p.quads, q);
        return;
    }

    Vec3d normals[] =
    {
        Normalize(q.verts[0].p + q.verts[1].p),
        Normalize(q.verts[0].p + q.verts[2].p),
        Normalize(q.verts[0].p + q.verts[1].p + q.verts[2].p + q.verts[3].p),
        Normalize(q.verts[1].p + q.verts[3].p),
        Normalize(q.verts[2].p + q.verts[3].p),
    };

#define VERT(n) { normals[n] * p.radius, normals[n] }
#define QUAD(a, b, c, d) {verts[a], verts[b], verts[c], verts[d]}

    QVert verts[] =
    {
        q.verts[0], VERT(0), q.verts[1],
        VERT(1),    VERT(2), VERT(3),
        q.verts[2], VERT(4), q.verts[3],
    };

    Quad children[] =
    {
        QUAD(0, 1, 3, 4),
        QUAD(1, 2, 4, 5),
        QUAD(3, 4, 6, 7),
        QUAD(4, 5, 7, 8),
    };

#undef VERT
#undef QUAD

    for (int i = 0; i < 4; i++)
    {
        ProcessQuad(p, children[i], depth + 1);
    }
}

void RenderPlanet(Planet &p, const Mat4 &proj, const Mat4 &view,
                  Vec3d eye_position)
{
    Vec3d normals[] =
    {
        Normalize(V3d(-1, -1, -1)), // 0
        Normalize(V3d( 1, -1, -1)), // 1
        Normalize(V3d( 1,  1, -1)), // 2
        Normalize(V3d(-1,  1, -1)), // 3
        Normalize(V3d(-1, -1,  1)), // 4
        Normalize(V3d( 1, -1,  1)), // 5
        Normalize(V3d( 1,  1,  1)), // 6
        Normalize(V3d(-1,  1,  1)), // 7
    };

#define VERT(n) { normals[n] * p.radius, normals[n] }
#define QUAD(a, b, c, d) {{VERT(a), VERT(b), VERT(d), VERT(c)}}

    Quad root_quads[] =
    {
        QUAD(0, 1, 2, 3), // front
        QUAD(1, 5, 6, 2), // right
        QUAD(5, 4, 7, 6), // back
        QUAD(4, 0, 3, 7), // left
        QUAD(3, 2, 6, 7), // top
        QUAD(4, 5, 1, 0), // bottom
    };

#undef VERT
#undef QUAD

    ListResize(p.quads, 0);

    int num_roots = ArrayCount(root_quads);
    for (int i = 0; i < num_roots; i++)
    {
        ProcessQuad(p, root_quads[i], 1);
    }

    p.uniforms.proj.value.m4 = proj;
    p.uniforms.view.value.m4 = view;

    for (int i = 0; i < p.quads.num; ++i)
    {
        Quad &q = p.quads.data[i];
        for (int j = 0; j < 4; ++j)
        {
            Vec3d e = q.verts[j].p - eye_position;
            Vec3d n = q.verts[j].n;
            p.uniforms.p[j].value.v3 = V3(e.x, e.y, e.z);
            p.uniforms.n[j].value.v3 = V3(n.x, n.y, n.z);
        }

        Draw(p.patch);
    }
}


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
        proj = Mat4PerspectiveLH(fovy, aspect, 1.0f, 1000.0f);
    }

    struct
    {
        Vec3d position;
        Vec3 angles;
    } cam = {};

    cam.position.z = -200.0;

    Planet planet = {};
    if (!InitPlanet(planet, 100.0))
    {
        return 1;
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

        double dt;
        {
            Uint32 ticks = SDL_GetTicks();
            Uint32 delta = ticks - last_t;
            last_t = ticks;
            dt = delta / 1000.0;
        }

        double move_speed = 30.0; // m/s
        float look_speed = 2.0f; // rad/s

        if (keys[SDL_SCANCODE_LSHIFT])
            move_speed = 100.0;

        Vec3d move = V3d(keys[SDL_SCANCODE_D] - keys[SDL_SCANCODE_A], 0,
                         keys[SDL_SCANCODE_W] - keys[SDL_SCANCODE_S]);
        Vec3 look = V3(keys[SDL_SCANCODE_DOWN] - keys[SDL_SCANCODE_UP],
                       keys[SDL_SCANCODE_RIGHT] - keys[SDL_SCANCODE_LEFT], 0);

        cam.angles += look * look_speed * dt;

        Mat3 rotation = (Mat3RotationY(cam.angles.y) *
                         Mat3RotationX(cam.angles.x) *
                         Mat3RotationZ(cam.angles.z));

        cam.position += (V3d(rotation[0]) * move.x +
                         V3d(rotation[1]) * move.y +
                         V3d(rotation[2]) * move.z) * move_speed * dt;

        Mat4 view = Mat4ComposeTR(rotation, V3(0.0f));
        view = Mat4InverseTR(view);

        // Render
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);

        RenderPlanet(planet, proj, view, cam.position);

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

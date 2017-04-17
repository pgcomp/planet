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

struct Quad
{
    Vec3d p[4];
    int lod;
};

struct Planet
{
    double radius;
    int max_lod;
    DrawItem patch;
    Texture hmap;
    struct
    {
        Uniform proj;
        Uniform view;
        Uniform p[4];
        Uniform n[4];
        Uniform color;
    } uniforms;
    List<Quad> quads;
};

bool InitPlanet(Planet &p, double radius)
{
    const char *shader_source =
        GLSL(ATTRIBUTE(vec2, UV, 0);

             SAMPLER(sampler2D, HeightMap, 0);

             uniform mat4 Projection;
             uniform mat4 View;
             uniform vec3 P[4];
             uniform vec3 N[4];

             struct V { vec3 p; vec3 n; };

             V interpolate_linear(V v0, V v1, float t) {
             vec3 n = normalize(mix(v0.n, v1.n, t));
             vec3 p = mix(v0.p, v1.p, t);

             V result;
             result.p = p;
             result.n = n;
             return result;
             }

             V interpolate(V v0, V v1, float t) {
             if (1.0 - dot(v0.n, v1.n) < 0.001)
             return interpolate_linear(v0, v1, t);

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
             float h = texture(HeightMap, UV).r * 10.0;
             v.p += h*v.n;
             gl_Position = Projection * View * vec4(v.p, 1.0);
             },

             uniform vec3 Color;
             void main() {
             gl_FragColor = vec4(Color, 1.0);
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

    const int tex_size = patch_size_in_verts + 2;
    const int tex_data_size = Square(tex_size);
    float tex_data[tex_data_size];
    for (int i = 0; i < tex_data_size; i++)
    {
        tex_data[i] = Sin(i*0.31f + 0.121f) + Cos(i*1.91f + 5.02f);
    }

    GLuint height_map = CreateTexture2D(tex_size, tex_size,
                                        GL_RED, GL_FLOAT, tex_data);
    p.hmap.texture = height_map;
    p.hmap.bind_index = 0;

    InitUniform(p.uniforms.proj, shader, "Projection", Mat4Identity());
    InitUniform(p.uniforms.view, shader, "View", Mat4Identity());
    for (int i = 0; i < 4; ++i)
    {
        char P[] = "P[_]"; P[2] = '0' + i;
        char N[] = "N[_]"; N[2] = '0' + i;
        InitUniform(p.uniforms.p[i], shader, P, V3(0.0f));
        InitUniform(p.uniforms.n[i], shader, N, V3(0.0f));
    };
    InitUniform(p.uniforms.color, shader, "Color", V3(0.0f));

    p.radius = radius;

    // l = log_2(2*pi*r/q) - 2
    p.max_lod = Log2(2.0*PI*radius/patch_size_in_quads) - 2;

    DrawItem &d = p.patch;
    d.shader = shader;
    d.vertex_array = vertex_array;
    d.textures = &p.hmap;
    d.texture_count = 1;
    d.uniforms = &p.uniforms.proj;
    d.uniform_count = GetUniformCountFromSize(sizeof(p.uniforms));
    d.primitive_mode = GL_TRIANGLE_STRIP;
    d.index_type = GL_UNSIGNED_INT;
    d.first = 0;
    d.count = ArrayCount(indices);
    d.draw_arrays = false;

    return true;
}

struct CameraInfo
{
    Vec3d position;
    Mat3 rotation;
    float proj_factor;
    float aspect_ratio;
    float near_plane;
    float far_plane;
};

void InitCameraInfo(CameraInfo &info, float fovy_rad, float aspect_ratio,
                    float near_plane, float far_plane)
{
    info.proj_factor = 1.0f/Tan(0.5*fovy_rad);
    info.aspect_ratio = aspect_ratio;
    info.near_plane = near_plane;
    info.far_plane = far_plane;
}

void ProcessQuad(Planet &p, const Quad &q, const CameraInfo &cam, int lod)
{
    if (lod == 0)
    {
        // No more splitting
        Quad Q = q;
        Q.lod = lod;
        ListAdd(p.quads, Q);
        return;
    }

    Vec3d mid = Normalize(q.p[0] + q.p[1] + q.p[2] + q.p[3]) * p.radius;

    // Let's assume the quad is a screen aligned square.
    // s = side of the square
    // pf = 1.0/Tan(fovy)
    // ar = aspect ratio
    // z = distance to the camera
    // The area of the NDC square is
    // A = (s*pf/ar/z) * (s*pf/z) = (s^2)*(pf^2/ar)/(z^2)

    // Diagonals
    Vec3d d1 = q.p[3] - q.p[0];
    Vec3d d2 = q.p[2] - q.p[1];

    // Diagonal is about sqrt(2) times the side.
    double avg_diagonal_sq = (LengthSq(d1) + LengthSq(d2)) * 0.5;
    double side_sq = avg_diagonal_sq * 0.5; // (diag/sqrt(2))^2 = (diag^2)/2
    double dist_sq = LengthSq(mid - cam.position);
    double f = Square(cam.proj_factor) / cam.aspect_ratio;

    double ndc_area = side_sq*f / dist_sq;
    double ndc_screen_area = Square(2.0);

    if (ndc_area < 0.5*ndc_screen_area)
    {
        // No need to split
        Quad Q = q;
        Q.lod = lod;
        ListAdd(p.quads, Q);
        return;
    }

    // Do split

#define VERT(i, j) Normalize(q.p[i] + q.p[j]) * p.radius
#define QUAD(a, b, c, d) (Quad){verts[a], verts[b], verts[c], verts[d]}

    Vec3d verts[] =
    {
        q.p[0], VERT(0, 1), q.p[1],
        VERT(0, 2), mid, VERT(1, 3),
        q.p[2], VERT(2, 3), q.p[3],
    };

    ProcessQuad(p, QUAD(0, 1, 3, 4), cam, lod - 1);
    ProcessQuad(p, QUAD(1, 2, 4, 5), cam, lod - 1);
    ProcessQuad(p, QUAD(3, 4, 6, 7), cam, lod - 1);
    ProcessQuad(p, QUAD(4, 5, 7, 8), cam, lod - 1);

#undef VERT
#undef QUAD
}

void RenderPlanet(Planet &p, const CameraInfo &cam)
{
    ListResize(p.quads, 0);

#define VERT(x, y, z) Normalize(V3d(x, y, z)) * p.radius
#define QUAD(a, b, c, d) (Quad){verts[a], verts[b], verts[d], verts[c]}

    Vec3d verts[] =
    {
        VERT(-1, -1, -1), // 0
        VERT( 1, -1, -1), // 1
        VERT( 1,  1, -1), // 2
        VERT(-1,  1, -1), // 3
        VERT(-1, -1,  1), // 4
        VERT( 1, -1,  1), // 5
        VERT( 1,  1,  1), // 6
        VERT(-1,  1,  1), // 7
    };

    ProcessQuad(p, QUAD(0, 1, 2, 3), cam, p.max_lod); // front
    ProcessQuad(p, QUAD(1, 5, 6, 2), cam, p.max_lod); // right
    ProcessQuad(p, QUAD(5, 4, 7, 6), cam, p.max_lod); // back
    ProcessQuad(p, QUAD(4, 0, 3, 7), cam, p.max_lod); // left
    ProcessQuad(p, QUAD(3, 2, 6, 7), cam, p.max_lod); // top
    ProcessQuad(p, QUAD(4, 5, 1, 0), cam, p.max_lod); // bottom

#undef VERT
#undef QUAD

    Mat4 projection = {};
    {
        double f = cam.far_plane;
        double n = cam.near_plane;

        projection[0][0] = cam.proj_factor/cam.aspect_ratio;
        projection[1][1] = cam.proj_factor;
        projection[2][2] = (f + n) / (f - n);
        projection[2][3] = 1.0;
        projection[3][2] = -2.0*f*n / (f - n);
    }

    Mat4 view = {};
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
            view[i][j] = cam.rotation[j][i];
    }
    view[3][3] = 1.0f;

    p.uniforms.proj.value.m4 = projection;
    p.uniforms.view.value.m4 = view;

    Vec3 colors[] =
    {
        /*  0 */ V3(1.0f),
        /*  1 */ V3(0.0f, 1.0f, 1.0f),
        /*  2 */ V3(1.0f, 0.0f, 1.0f),
        /*  3 */ V3(1.0f, 1.0f, 0.0f),
        /*  4 */ V3(0.0f, 0.0f, 1.0f),
        /*  5 */ V3(0.0f, 1.0f, 0.0f),
        /*  6 */ V3(1.0f, 0.0f, 0.0f),

        /*  7 */ V3(0.5f),
        /*  8 */ V3(0.0f, 0.5f, 0.5f),
        /*  9 */ V3(0.5f, 0.0f, 0.5f),
        /* 10 */ V3(0.5f, 0.5f, 0.0f),
        /* 11 */ V3(0.0f, 0.0f, 0.5f),
        /* 12 */ V3(0.0f, 0.5f, 0.0f),
        /* 13 */ V3(0.5f, 0.0f, 0.0f),

        /* 14 */ V3(0.25f),
        /* 15 */ V3(0.5f, 0.0f, 1.0f),
        /* 16 */ V3(0.0f, 1.0f, 0.5f),
        /* 17 */ V3(1.0f, 0.5f, 0.0f),
        /* 18 */ V3(0.0f, 0.5f, 1.0f),
        /* 19 */ V3(0.5f, 1.0f, 0.0f),
        /* 20 */ V3(1.0f, 0.0f, 0.5f),
    };

    for (int i = 0; i < p.quads.num; ++i)
    {
        Quad &q = p.quads.data[i];

        for (int j = 0; j < 4; ++j)
        {
            Vec3d e = q.p[j] - cam.position;
            Vec3d n = Normalize(q.p[j]);
            p.uniforms.p[j].value.v3 = V3(e.x, e.y, e.z);
            p.uniforms.n[j].value.v3 = V3(n.x, n.y, n.z);
        }

        p.uniforms.color.value.v3 = colors[q.lod];

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

    double radius = 6371000.0;

    Planet planet = {};
    if (!InitPlanet(planet, radius))
    {
        return 1;
    }

    struct
    {
        Vec3d position;
        Vec3 angles;
    } cam = {};

    cam.position.z = -radius - 10.0;

    double move_speed = 1000.0; // m/s
    float look_speed = 2.0f; // rad/s

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
                switch (event.key.keysym.scancode)
                {
                    case SDL_SCANCODE_1: move_speed = 1.0e1; break;
                    case SDL_SCANCODE_2: move_speed = 1.0e2; break;
                    case SDL_SCANCODE_3: move_speed = 1.0e3; break;
                    case SDL_SCANCODE_4: move_speed = 1.0e4; break;
                    case SDL_SCANCODE_5: move_speed = 1.0e5; break;
                    case SDL_SCANCODE_6: move_speed = 1.0e6; break;
                    case SDL_SCANCODE_7: move_speed = 1.0e7; break;
                    case SDL_SCANCODE_8: move_speed = 1.0e8; break;
                    case SDL_SCANCODE_9: move_speed = 1.0e9; break;
                    case SDL_SCANCODE_0: move_speed = 1.0e10; break;
                    default: break;
                }
            }
        }

        const Uint8 *keys = SDL_GetKeyboardState(nullptr);

        double dt;
        int dt_ms;
        {
            Uint32 ticks = SDL_GetTicks();
            Uint32 delta = ticks - last_t;
            last_t = ticks;
            dt = delta / 1000.0;
            dt_ms = delta;
        }

        {
            int tri_count = planet.quads.num * 29*29*2;

            char title[100];
            snprintf(title, sizeof(title),
                     "frametime: %d, fps: %d, tris: %d",
                     dt_ms, (int)(1.0/dt), tri_count);
            SDL_SetWindowTitle(window, title);
        }

        Vec3d move = V3d(keys[SDL_SCANCODE_D] - keys[SDL_SCANCODE_A], 0,
                         keys[SDL_SCANCODE_W] - keys[SDL_SCANCODE_S]);
        Vec3 look = V3(keys[SDL_SCANCODE_DOWN] - keys[SDL_SCANCODE_UP],
                       keys[SDL_SCANCODE_RIGHT] - keys[SDL_SCANCODE_LEFT], 0);

        cam.angles += look * look_speed * dt;

        Vec3 up = V3(Normalize(cam.position));
        Vec3 right;
        if (1.0f - Dot(up, V3(0.0f, 1.0f, 0.0f)) < 0.1f)
            right = Normalize(Cross(up, V3(0.0f, 0.0f, 1.0f)));
        else
            right = Normalize(Cross(up, V3(0.0f, 1.0f, 0.0f)));
        Vec3 forward = Normalize(Cross(right, up));

        Mat3 base_rotation = Mat3FromBasisVectors(right, up, forward);

        // TODO: 6 degrees of freedom
        Mat3 rotation = (Mat3RotationY(cam.angles.y) *
                         Mat3RotationX(cam.angles.x) *
                         Mat3RotationZ(cam.angles.z));

        rotation = base_rotation * rotation;

        cam.position += (V3d(rotation[0]) * move.x +
                         V3d(rotation[1]) * move.y +
                         V3d(rotation[2]) * move.z) * move_speed * dt;

        float fovy = DegToRad(50.0f);
        float aspect = (float)window_w/window_h;
        float near = 1.0f;
        float far = 20000000.0f;

        CameraInfo cam_info;
        cam_info.position = cam.position;
        cam_info.rotation = rotation;
        cam_info.proj_factor = 1.0f/Tan(0.5f*fovy);
        cam_info.aspect_ratio = aspect;
        cam_info.near_plane = near;
        cam_info.far_plane = far;

        // Render
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);

        RenderPlanet(planet, cam_info);

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

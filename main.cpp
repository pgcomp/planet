#include <SDL2/SDL.h>
#include "render.h"
#include "list.h"
#include "logging.h"
#include "timing.h"
#include "pp.h"

#if 0
extern "C"
{
    __attribute__ ((dllexport)) uint32_t NvOptimusEnablement = 0x00000001;
    __attribute__ ((dllexport)) int AmdPowerXpressRequestHighPerformance = 1;
}
#endif


#define U64(x) ((uint64_t)(x))

struct QuadID
{
    uint64_t value;
};

#define GET_BITS(value, first, count) (((value)>>U64(first)) & ((U64(1)<<U64(count))-U64(1)))

inline uint64_t GetRoot(QuadID id)  { return GET_BITS(id.value, 64 - 4, 3); }
inline uint64_t GetDepth(QuadID id) { return GET_BITS(id.value, 64 - 9, 5); }
inline uint64_t GetIndex(QuadID id) { return GET_BITS(id.value, 0, 64 - 9); }

#undef GET_BITS

inline QuadID MakeRootID(uint64_t root)
{
    assert(root < U64(6));
    // only valid ids have highest bit set -> zero id is invalid, yay!
    uint64_t highest_bit = U64(1) << U64(64 - 1);
    uint64_t value = highest_bit | (root << U64(64 - 4));
    return {value};
}

inline QuadID MakeChildID(QuadID id, uint64_t child_index)
{
    uint64_t depth = GetDepth(id);
    assert(child_index < U64(4));
    assert(depth + U64(1) < U64(32));
    uint64_t depth_bit = U64(1) << U64(64 - 9);
    uint64_t value = (id.value + depth_bit) | (child_index << U64(2*depth));
    return {value};
}

inline uint64_t GetChildIndex(QuadID id)
{
    uint64_t depth = GetDepth(id);
    return (id.value >> U64(2*(depth - 1))) & U64(3);
}

inline QuadID GetParentID(QuadID id)
{
    uint64_t depth = GetDepth(id);
    assert(depth != 0);
    uint64_t depth_bit = U64(1) << U64(64 - 9);
    uint64_t mask = ~(U64(3) << U64(2*(depth - 1)));
    uint64_t value = (id.value - depth_bit) & mask;
    return {value};
}


struct Quad
{
    Vec3d p[4];
    QuadID id;
};


#define CACHE_MAX 1024 //512
#define MAP_MAX 1499 //661 // prime

struct HeightMapCache
{
    int count;
    QuadID quad_ids[MAP_MAX];
    GLuint height_maps[MAP_MAX];
    uint32_t last_tick_used[MAP_MAX];
};

int MapFind(const HeightMapCache &cache, QuadID key, QuadID find)
{
    assert(key.value != U64(0));

    // TODO: Better hash
    uint32_t hash = uint32_t(key.value) ^ uint32_t(key.value >> 32);
    for (int i = 0; i < MAP_MAX; i++)
    {
        int index = (hash + i) % MAP_MAX;
        if (cache.quad_ids[index].value == find.value)
        {
            return index;
        }
    }

    return -1;
}

#undef U64


struct HeightMapGenerator
{
    float (*GetHeightAt)(const Vec3d &, int, int);
    void (*GenerateHeightMap)(float *, int, const Quad &, int);
};

template<typename F>
HeightMapGenerator CreateHeightMapGenerator()
{
    struct Gen
    {
        static float GetHeightAt(const Vec3d &p, int depth, int max_depth)
        {
            return F()(p, depth, max_depth);
        }

        static void GenerateHeightMap(float *data, int dim, const Quad &q,
                                      int max_depth)
        {
            TIMED_FUNCTION();

            assert(dim > 3);

            int depth = GetDepth(q.id);

            Vec3d v0 = q.p[1] - q.p[0];
            Vec3d v1 = q.p[3] - q.p[2];

            double div = 1.0/(dim - 3);
            for (int y = 0; y < dim; y++)
            {
                for (int x = 0; x < dim; x++)
                {
                    double u = (x - 1)*div;
                    double v = (y - 1)*div;

                    Vec3d p0 = q.p[0] + v0 * u;
                    Vec3d p1 = q.p[2] + v1 * u;
                    Vec3d v2 = p1 - p0;
                    Vec3d p = p0 + v2 * v;

                    data[y*dim + x] = F()(p, depth, max_depth);
                }
            }
        }
    };

    HeightMapGenerator result;
    result.GetHeightAt = Gen::GetHeightAt;
    result.GenerateHeightMap = Gen::GenerateHeightMap;
    return result;
};


struct Planet
{
    double radius;
    int max_lod;
    float max_skirt_size;
    DrawItem patch;
    Texture hmap;
    struct
    {
        Uniform proj;
        Uniform view;
        Uniform p;
        Uniform n;
        Uniform skirt_size;
        TexUniforms hmap;
    } uniforms;
    List<Quad> quads;
    HeightMapCache cache;
    HeightMapGenerator hmap_gen;
};


struct TextureRect
{
    GLuint texture;
    Vec2 corners[2];
    Vec2 pixel_size;
};

TextureRect GetHeightMapForQuad(Planet &planet, const Quad &q, int tick,
                                int &generations_per_frame_left)
{
    const int dim = 32;

    TextureRect result = {};
    result.corners[0] = V2(1.5f / dim);
    result.corners[1] = V2((dim - 1.5f) / dim);
    result.pixel_size = V2(1.0f / dim);

    HeightMapCache &cache = planet.cache;

    int index = MapFind(cache, q.id, q.id);

    if (index < 0)
    {
        int depth = GetDepth(q.id);
        if (generations_per_frame_left <= 0 && depth > 0)
        {
            QuadID parent_id = GetParentID(q.id);
            index = MapFind(cache, parent_id, parent_id);
            if (index >= 0)
            {
                int child_index = GetChildIndex(q.id);

                float x0 = 1.5f;
                float y0 = x0;
                float x1 = dim / 2.0f - 0.5f;
                float y1 = x1;

                if (child_index == 1 || child_index == 3)
                {
                    x0 = dim / 2.0f + 0.5f;
                    x1 = dim - 1.5f;
                }

                if (child_index == 2 || child_index == 3)
                {
                    y0 = dim / 2.0f + 0.5f;
                    y1 = dim - 1.5f;
                }

                result.corners[0] = V2(x0 / dim, y0 / dim);
                result.corners[1] = V2(x1 / dim, y1 / dim);
                result.pixel_size = V2(((dim / 2.0f - 1.0f) / (dim - 3)) / dim);
            }
        }

        if (generations_per_frame_left > 0 || index < 0)
        {
            generations_per_frame_left--;

            float data[dim*dim];
#if 1
            planet.hmap_gen.GenerateHeightMap(data, dim, q, planet.max_lod);
#else
            BEGIN_TIMED_BLOCK(GenerateHeightMap);

            assert(dim > 3);

            int depth = GetDepth(q.id);

            Vec3d v0 = q.p[1] - q.p[0];
            Vec3d v1 = q.p[3] - q.p[2];

            double div = 1.0/(dim - 3);
            for (int y = 0; y < dim; y++)
            {
                for (int x = 0; x < dim; x++)
                {
                    double u = (x - 1)*div;
                    double v = (y - 1)*div;

                    Vec3d p0 = q.p[0] + v0 * u;
                    Vec3d p1 = q.p[2] + v1 * u;
                    Vec3d v2 = p1 - p0;
                    Vec3d p = p0 + v2 * v;

                    //data[y*dim + x] = planet.hmap_gen.GetHeightAt(p, depth, planet.max_lod);
                    data[y*dim + x] = GenPerlinRidge(p, depth, planet.max_lod);
                }
            }

            END_TIMED_BLOCK(GenerateHeightMap);
#endif
            GLuint height_map = CreateTexture2D(dim, dim, GL_RED, GL_FLOAT, data);

            if (cache.count == CACHE_MAX)
            {
                int lru;
                int delta_ticks = -1;

                for (int i = 0; i < MAP_MAX; i++)
                {
                    uint32_t t = cache.last_tick_used[i];
                    int delta = tick - t;
                    if (t != 0 && delta > delta_ticks)
                    {
                        lru = i;
                        delta_ticks = delta;
                    }
                }

                DeleteTexture(cache.height_maps[lru]);
                cache.quad_ids[lru] = QuadID{0};
                cache.last_tick_used[lru] = 0;
                cache.count--;
            }

            index = MapFind(cache, q.id, QuadID{0});
            cache.quad_ids[index] = q.id;
            cache.height_maps[index] = height_map;
            cache.count++;
        }
    }

    if (tick == 0) tick = 1; // zero means not in use
    cache.last_tick_used[index] = tick;
    result.texture = cache.height_maps[index];
    return result;
}

bool InitPlanet(Planet &p, double radius, HeightMapGenerator hmap_gen)
{
    const char *shader_source = R"GLSL(

        VARYING(vec3, Normal);

#if VERTEX_SHADER

    ATTRIBUTE(vec3, UV, 0);

    SAMPLER(sampler2D, HeightMap, 0);

    uniform mat4 Projection;
    uniform mat4 View;
    uniform vec3 P[4];
    uniform vec3 N[4];
    uniform float SkirtSize;

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

    float sample_height(vec2 uv) {
        return texture(HeightMap, uv).r;
    }

    vec3 compute_normal(vec2 uv, float xyscale) {
        vec3 offs = vec3(HeightMap_pixel_size.x, 0.0,
                         HeightMap_pixel_size.y);
        float x0 = sample_height(uv - offs.xy);
        float x1 = sample_height(uv + offs.xy);
        float y0 = sample_height(uv - offs.yz);
        float y1 = sample_height(uv + offs.yz);
        return normalize(vec3(x0 - x1, 2.0*xyscale, y0 - y1));
    }

    void main() {
        V a; a.p = P[0]; a.n = N[0];
        V b; b.p = P[1]; b.n = N[1];
        V c; c.p = P[2]; c.n = N[2];
        V d; d.p = P[3]; d.n = N[3];

        V p = interpolate(a, b, UV.x);
        V q = interpolate(c, d, UV.x);
        V v = interpolate(p, q, UV.y);

        vec2 uv = mix(HeightMap_corners[0], HeightMap_corners[1], UV.xy);

        float height = sample_height(uv) - SkirtSize*UV.z;
        vec3 normal = compute_normal(uv, length(q.p - p.p) / 29.0);
        vec3 n = v.n;
        vec3 t = normalize(cross(n, q.p - p.p));
        vec3 bi = normalize(cross(t, n));
        Normal = normalize(mat3(t, n, bi) * normal);
        gl_Position = Projection * View * vec4(v.p + v.n*height, 1.0);
    }

#elif FRAGMENT_SHADER

    out vec4 FragColor;

    void main() {
        vec3 n = normalize(Normal);
        vec3 l = normalize(vec3(0.0, 1.0, -1.0));
        float light = 0.001 + max(0.0, dot(n, l));
        //FragColor = vec4(1.0);
        FragColor = vec4(vec3(sqrt(light)), 1.0);
        //FragColor = vec4(n * 0.5 + vec3(0.5), 1.0);
    }

#endif)GLSL";
    // END_GLSL

    GLuint shader = CreateShaderFromSource(shader_source);
    if (shader == 0)
    {
        return false;
    }

    const int patch_size_in_verts = 30; // configurable
    const int patch_size_in_quads = patch_size_in_verts - 1;
    const int skirt_verts = 4*patch_size_in_verts;
    const int verts_total = Square(patch_size_in_verts) + skirt_verts;
    const int indices_per_reset = 2;
    const int indices_per_strip =
        2 + patch_size_in_quads*2 + indices_per_reset;
    const int skirt_indices = patch_size_in_quads*4 + 2*indices_per_strip;
    const int indices_total =
        patch_size_in_quads*indices_per_strip - indices_per_reset + skirt_indices;

    Vec3 verts[verts_total];
    {
        int n = 0;

        double div = 1.0 / patch_size_in_quads;

        for (int x = 0; x < patch_size_in_verts; ++x)
            verts[n++] = V3(x*div, 0.0f, 1.0f); // skirt

        for (int y = 0; y < patch_size_in_verts; ++y)
        {
            verts[n++] = V3(0.0f, y*div, 1.0f); // skirt

            for (int x = 0; x < patch_size_in_verts; ++x)
                verts[n++] = V3(x*div, y*div, 0.0f);

            verts[n++] = V3(1.0f, y*div, 1.0f); // skirt
        }

        for (int x = 0; x < patch_size_in_verts; ++x)
            verts[n++] = V3(x*div, 1.0f, 1.0f); // skirt

        assert(n == verts_total);
    }

    uint32_t indices[indices_total];
    {
        int n = 0;

        uint32_t v0 = 0;
        uint32_t v1 = patch_size_in_verts + 1;

        // skirt
        for (int x = 0; x < patch_size_in_verts; ++x)
        {
            indices[n++] = v0++;
            indices[n++] = v1++;
        }

        // reset
        indices[n++] = v1-1;
        indices[n++] = v0;
        v1++;

        for (int y = 0; y < patch_size_in_quads; ++y)
        {
            for (int x = 0; x < patch_size_in_verts + 2; ++x)
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

        v0++;
        // reset
        indices[n++] = v1-1;
        indices[n++] = v0;

        // skirt
        for (int x = 0; x < patch_size_in_verts; ++x)
        {
            indices[n++] = v0++;
            indices[n++] = v1++;
        }

        assert(n == indices_total);
    }

    VertexFormat vf = {};
    AddVertexAttribute(vf, 0, 3, GL_FLOAT, false);

    GLuint vertex_buffer = CreateVertexBuffer(sizeof(verts), verts);
    GLuint index_buffer = CreateIndexBuffer(sizeof(indices), indices);
    GLuint vertex_array = CreateVertexArray(vertex_buffer, vf, index_buffer);

    p.hmap.texture = 0;
    p.hmap.bind_index = 0;

    InitTexUniforms(p.uniforms.hmap, shader, "HeightMap");

    InitUniform(p.uniforms.proj, shader, "Projection", Uniform::MAT4);
    InitUniform(p.uniforms.view, shader, "View", Uniform::MAT4);
    InitUniform(p.uniforms.p, shader, "P", Uniform::VEC3, 4);
    InitUniform(p.uniforms.n, shader, "N", Uniform::VEC3, 4);
    InitUniform(p.uniforms.skirt_size, shader, "SkirtSize", Uniform::FLOAT);

    p.radius = radius;
    p.hmap_gen = hmap_gen;

    // l = log_2(2*pi*r/q) - 2
    p.max_lod = Log2(2.0*PI*radius/patch_size_in_quads) - 2;

    // s = 2*pi*r/(4*q)*p_scale*8*h_scale
    p.max_skirt_size = (2*PI*radius)/(4*patch_size_in_quads)*0.00001*8*8848.0;

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
    d.count = indices_total;
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

    // adhock stuff
    Vec3d real_pos;
    bool draw_nodes;
};

void InitCameraInfo(CameraInfo &info, float fovy_rad, float aspect_ratio,
                    float near_plane, float far_plane)
{
    info.proj_factor = 1.0f/Tan(0.5*fovy_rad);
    info.aspect_ratio = aspect_ratio;
    info.near_plane = near_plane;
    info.far_plane = far_plane;
}

void ProcessQuad(Planet &planet, const Quad &q, const CameraInfo &cam, int lod)
{
    if (lod == 0)
    {
        // No more splitting
        ListAdd(planet.quads, q);
        return;
    }

    Vec3d mid_n = Normalize(q.p[0] + q.p[1] + q.p[2] + q.p[3]);
    Vec3d mid = mid_n * planet.radius;

    Vec3d p[5];
    for (int i = 0; i < 4; i++)
    {
        float height = planet.hmap_gen.GetHeightAt(q.p[i], 0, 1);
        p[i] = q.p[i] + Normalize(q.p[i]) * height;
    }
    float height = planet.hmap_gen.GetHeightAt(mid, 0, 1);
    p[4] = mid + mid_n * height;

    bool split = false;

    double d =
        (LengthSq(p[3] - p[0]) + LengthSq(p[2] - p[1]))
        / (1.0 + 2.5*lod / planet.max_lod);

    for (int i = 0; i < 5; i++)
    {
        if (LengthSq(p[i] - cam.position)*2.0 < d)
        {
            split = true;
            break;
        }
    }

    if (!split)
    {
        ListAdd(planet.quads, q);
        return;
    }

    // Do split

#define VERT(i, j) Normalize(q.p[i] + q.p[j]) * planet.radius
#define QUAD(a, b, c, d, id) (Quad){verts[a], verts[b], verts[c], verts[d], id}

    Vec3d verts[] =
    {
        q.p[0], VERT(0, 1), q.p[1],
        VERT(0, 2), mid, VERT(1, 3),
        q.p[2], VERT(2, 3), q.p[3],
    };

    ProcessQuad(planet, QUAD(0, 1, 3, 4, MakeChildID(q.id, 0)), cam, lod - 1);
    ProcessQuad(planet, QUAD(1, 2, 4, 5, MakeChildID(q.id, 1)), cam, lod - 1);
    ProcessQuad(planet, QUAD(3, 4, 6, 7, MakeChildID(q.id, 2)), cam, lod - 1);
    ProcessQuad(planet, QUAD(4, 5, 7, 8, MakeChildID(q.id, 3)), cam, lod - 1);

#undef VERT
#undef QUAD
}

void RenderPlanet(Planet &planet, CameraInfo cam)
{
    ListResize(planet.quads, 0);

#define VERT(x, y, z) Normalize(V3d(x, y, z)) * planet.radius
#define QUAD(a, b, c, d, id) (Quad){verts[a], verts[b], verts[d], verts[c], id}

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

    ProcessQuad(planet, QUAD(0, 1, 2, 3, MakeRootID(0)), cam, planet.max_lod); // front
    ProcessQuad(planet, QUAD(1, 5, 6, 2, MakeRootID(1)), cam, planet.max_lod); // right
    ProcessQuad(planet, QUAD(5, 4, 7, 6, MakeRootID(2)), cam, planet.max_lod); // back
    ProcessQuad(planet, QUAD(4, 0, 3, 7, MakeRootID(3)), cam, planet.max_lod); // left
    ProcessQuad(planet, QUAD(3, 2, 6, 7, MakeRootID(4)), cam, planet.max_lod); // top
    ProcessQuad(planet, QUAD(4, 5, 1, 0, MakeRootID(5)), cam, planet.max_lod); // bottom

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

    planet.uniforms.proj.value.m4 = projection;
    planet.uniforms.view.value.m4 = view;

#if 0
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
#endif

    // Allows us to "freeze" LOD
    Vec3d vantage_pos = cam.position;
    cam.position = cam.real_pos;

    // TODO: Fix
    static uint32_t tick = 0;
    tick++;

    int generations_per_frame = 100;

    //if (!cam.draw_nodes)
    //{
    for (int i = 0; i < planet.quads.num; ++i)
    {
        Quad &q = planet.quads.data[i];

        TextureRect texrect = GetHeightMapForQuad(planet, q, tick,
                                                  generations_per_frame);
        planet.hmap.texture = texrect.texture;
        planet.uniforms.hmap.corners.value.v2[0] = texrect.corners[0];
        planet.uniforms.hmap.corners.value.v2[1] = texrect.corners[1];
        planet.uniforms.hmap.pixel_size.value.v2[0] = texrect.pixel_size;

        for (int j = 0; j < 4; ++j)
        {
            Vec3d p = q.p[j] - cam.position;
            Vec3d n = Normalize(q.p[j]);
            planet.uniforms.p.value.v3[j] = V3(p.x, p.y, p.z);
            planet.uniforms.n.value.v3[j] = V3(n.x, n.y, n.z);
        }

        float skirt_size = planet.max_skirt_size;
        int depth = GetDepth(q.id) - 1;
        if (depth > 0) skirt_size /= (2 << depth);
        planet.uniforms.skirt_size.value.f[0] = skirt_size;

        Draw(planet.patch);
    }
    //}
#if 0
    else
    {
        glUseProgram(0);

        glMatrixMode(GL_PROJECTION);
        glLoadMatrixf(projection.data[0]);
        glMatrixMode(GL_MODELVIEW);
        glLoadMatrixf(view.data[0]);

        glBegin(GL_QUADS);
        glColor3f(1.0f, 1.0f, 1.0f);
        for (int i = 0; i < planet.quads.num; ++i)
        {
            Quad &q = planet.quads.data[i];
            int indices[4] = {2, 3, 1, 0}; //{0, 1, 3, 2};
            for (int j = 0; j < 4; ++j)
            {
                int index = indices[j];
                Vec3d p = q.p[index] - cam.position;
                glVertex3f(p.x, p.y, p.z);
            }
        }
        glEnd();
    }
    glUseProgram(0);

    glMatrixMode(GL_PROJECTION);
    glLoadMatrixf(projection.data[0]);
    glMatrixMode(GL_MODELVIEW);
    glLoadMatrixf(view.data[0]);

    glBegin(GL_POINTS);
    glColor3f(1.0f, 0.0f, 0.0f);
    Vec3d p = vantage_pos - cam.position;
    glVertex3f(p.x, p.y, p.z);
    glEnd();
#endif
}


#define PERLIN_IMPLEMENTATION
#include "perlin.h"

float PerlinfBm(double x, double y, double z, double lacunarity, float gain, int octaves)
{
    //double lacunarity = 2.0;
    //float gain = 0.5f;
    //int octaves = 6;

    double frequency = 1.0;
    float amplitude = 1.0f;
    float value = 0.0f;

    for (int i = 0; i < octaves; ++i)
    {
        value += PerlinNoise3(x*frequency, y*frequency, z*frequency)*amplitude;
        frequency *= lacunarity;
        amplitude *= gain;
    }

    return value;
}

float PerlinRidged(double x, double y, double z, double lacunarity, float gain, int octaves)
{
    //double lacunarity = 2.0;
    //float gain = 0.5f;
    float offset = 1.0f;
    //int octaves = 6;

    double frequency = 1.0;
    float amplitude = 1.0f;
    float weight = 1.0f;
    float value = 0.0f;

    for (int i = 0; i < octaves; ++i)
    {
        float v = PerlinNoise3(x*frequency, y*frequency, z*frequency);
        v = (v < 0.0f) ? -v : v;
        v = offset - v;
        v = v*v;
        value += v*amplitude*weight;
        weight = v;
        frequency *= lacunarity;
        amplitude *= gain;
    }

    return value;
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
                                          SDL_WINDOW_OPENGL|
                                          SDL_WINDOW_RESIZABLE);

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

    glFrontFace(GL_CW);
    glCullFace(GL_BACK);
    glEnable(GL_CULL_FACE);

    //glLineWidth(1.5f);
    //glPointSize(7.5f);

    bool wire_frame = false;
    bool freeze_lod = false;
    bool draw_nodes = false;
    Vec3d freeze_pos = {};
    float temp_skirt_size = 0.0f;

    double radius = 6371000.0;

    struct Perlin
    {
        inline float operator()(Vec3d p, int depth, int max_depth)
        {
            int octaves = 6 + 12 * depth / max_depth;
            p *= 0.00001;
            float h = PerlinRidged(p.x, p.y, p.z, 2.0f, 0.55f, octaves);
            //float h = PerlinfBm(p.x, p.y, p.z, 2.0f, 0.55f, octaves);
            return h * 8848.0f;
        }
    };

    struct ConstantZero
    {
        inline float operator()(Vec3d p, int depth, int max_depth)
        {
            return 0.0f;
        }
    };

    HeightMapGenerator hmap_gen = CreateHeightMapGenerator<Perlin>();
    //HeightMapGenerator hmap_gen = CreateHeightMapGenerator<ConstantZero>();

    Planet planet = {};
    if (!InitPlanet(planet, radius, hmap_gen))
    {
        return 1;
    }

    struct Camera
    {
        Vec3d position;
        Vec3 angles;
    };

    struct SaveState
    {
        Camera active_camera;
        Camera saved_cameras[12];
    };

    SaveState save = {};
    save.active_camera.position.z = -radius - 10.0;

    const char *save_file_name = "save";

    // Load state
    {
        FILE *f = fopen(save_file_name, "rb");
        if (f)
        {
            SaveState temp = {};
            size_t size = sizeof(SaveState);

            if (fread(&temp, 1, size, f) == size)
            {
                save = temp;
            }
            else
            {
                LOG_WARNING("Couldn't read save file.");
            }

            fclose(f);
        }
    }

    Camera cam = save.active_camera;

    cam.position = V3d(0.0);
    //cam.position.z = -(radius * 2.3);
    cam.position.z = -radius - 1000.0;
    cam.angles = V3(0.0f);
    //cam.angles.x = DegToRad(90.0f);

    //freeze_lod = true;

    Vec3d frz_p[] =
    {
        V3d(radius * 0.25, -(radius * 0.25), -(radius * 2.0)),
        V3d(radius * 0.25, -(radius * 0.25), -(radius * 1.5)),
        V3d(radius * 0.25, -(radius * 0.25), -(radius * 1.3)),
        V3d(radius * 0.25, -(radius * 0.25), -(radius * 1.2)),
        V3d(radius * 0.25, -(radius * 0.25), -(radius * 1.0)),
    };
    int frz_i = 0;
    int frz_mod = sizeof(frz_p) / sizeof(*frz_p);
    freeze_pos = frz_p[frz_i];

    double move_speed = 1000.0; // m/s
    float look_speed = 2.0f; // rad/s

    // Start looping
    //
    Uint32 last_t = SDL_GetTicks();
    bool running = true;

    while (running)
    {
        START_FRAME();

        // Handle events
        BEGIN_TIMED_BLOCK(EventHandling);

        for (SDL_Event event; SDL_PollEvent(&event); )
        {
            switch (event.type)
            {
                case SDL_QUIT:
                {
                    running = false;
                    break;
                }

                case SDL_WINDOWEVENT:
                {
                    switch(event.window.event)
                    {
                        case SDL_WINDOWEVENT_RESIZED:
                        {
                            window_w = event.window.data1;
                            window_h = event.window.data2;
                            glViewport(0, 0, window_w, window_h);
                            break;
                        }

                        default: break;
                    }

                    break;
                }

                case SDL_KEYDOWN:
                {
                    bool shift = event.key.keysym.mod & KMOD_SHIFT;

                    switch (event.key.keysym.scancode)
                    {
                        case SDL_SCANCODE_ESCAPE:
                        {
                            running = false;
                            break;
                        }

                        case SDL_SCANCODE_1: move_speed = 1.0e1; break;
                        case SDL_SCANCODE_2: move_speed = 1.0e2; break;
                        case SDL_SCANCODE_3: move_speed = 1.0e3; break;
                        case SDL_SCANCODE_4: move_speed = 1.0e4; break;
                        case SDL_SCANCODE_5: move_speed = 1.0e5; break;
                        case SDL_SCANCODE_6: move_speed = 1.0e6; break;
                        case SDL_SCANCODE_7: move_speed = 1.0e7; break;
                        case SDL_SCANCODE_8: move_speed = 1.0e8; break;
                        //case SDL_SCANCODE_9: move_speed = 1.0e9; break;
                        //case SDL_SCANCODE_0: move_speed = 1.0e10; break;

#define SAVE_CAM(index) case SDL_SCANCODE_F##index: { \
                            Camera &saved = save.saved_cameras[index-1]; \
                            if (shift) saved = cam; \
                            else if (Length(saved.position) > 0.001) cam = saved; \
                        } break

                        SAVE_CAM(1);
                        SAVE_CAM(2);
                        SAVE_CAM(3);
                        SAVE_CAM(4);
                        SAVE_CAM(5);
                        SAVE_CAM(6);
                        SAVE_CAM(7);
                        SAVE_CAM(8);
                        SAVE_CAM(9);
                        SAVE_CAM(10);
                        SAVE_CAM(11);
                        SAVE_CAM(12);

#undef SAVE_CAM

                        // Toggle wire frame
                        case SDL_SCANCODE_P:
                        {
                            wire_frame = !wire_frame;
                            glPolygonMode(GL_FRONT_AND_BACK, wire_frame ? GL_LINE : GL_FILL);
                            break;
                        }

                        // Toggle LOD freezing
                        case SDL_SCANCODE_L:
                        {
                            freeze_lod = !freeze_lod;
                            if (freeze_lod)
                                freeze_pos = cam.position;
                            break;
                        }

                        // Use next freeze pos
                        case SDL_SCANCODE_J:
                        {
                            frz_i = (frz_i + 1) % frz_mod;
                            freeze_pos = frz_p[frz_i];
                            break;
                        }

                        case SDL_SCANCODE_N:
                        {
                            draw_nodes = !draw_nodes;
                            break;
                        }

                        // Toggle skirts
                        case SDL_SCANCODE_K:
                        {
                            float temp = planet.max_skirt_size;
                            planet.max_skirt_size = temp_skirt_size;
                            temp_skirt_size = temp;
                            break;
                        }

                        case SDL_SCANCODE_T:
                        {
                            TOGGLE_TIMING();
                            break;
                        }

                        default: break;
                    }

                    break;
                }

                default: break;
            }
        }

        END_TIMED_BLOCK(EventHandling);


        BEGIN_TIMED_BLOCK(UpdateSimulation);

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
                     "frametime: %d, fps: %d, tris: %d, quads: %d",
                     dt_ms, (int)(1.0/dt), tri_count, planet.quads.num);
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

        END_TIMED_BLOCK(UpdateSimulation);


        BEGIN_TIMED_BLOCK(Rendering);

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

        cam_info.real_pos = cam.position;
        if (freeze_lod) cam_info.position = freeze_pos;
        cam_info.draw_nodes = draw_nodes;

        // Render
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);

        RenderPlanet(planet, cam_info);

        END_TIMED_BLOCK(Rendering);


        BEGIN_TIMED_BLOCK(DrawScreen);

        SDL_GL_SwapWindow(window);
        SDL_Delay(10);

        END_TIMED_BLOCK(DrawScreen);


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

    // Save state
    {
        save.active_camera = cam;

        FILE *f = fopen(save_file_name, "wb");
        if (f)
        {
            size_t size = sizeof(SaveState);

            if (fwrite(&save, 1, size, f) != size)
            {
                LOG_ERROR("Couldn't write save file.");
            }

            fclose(f);
        }
        else
        {
            LOG_ERROR("Couldn't open save file for writing.");
        }
    }

    // Done
    return 0;
}

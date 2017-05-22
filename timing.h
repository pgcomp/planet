#ifndef TIMING_H
#define TIMING_H

inline uint64_t GetMicroTicks()
{
    static Uint64 freq = SDL_GetPerformanceFrequency();
    return SDL_GetPerformanceCounter()*1000000ull / freq;
}

static bool print_timings = false;
static uint64_t frame_start_time = 0;

#define PRINT_TIMING(timer_name, timer_var) if (print_timings) printf("%-20s %10u us\n", timer_name, uint32_t(GetMicroTicks() - timer_var))

struct ScopeTimer
{
    const char *name;
    uint64_t start_time;
    ScopeTimer(const char *n) : name(n), start_time(GetMicroTicks()) {}
    ~ScopeTimer() { PRINT_TIMING(name, start_time); }
};

#define START_TIMING() print_timings = true
#define STOP_TIMING() print_timings = false
#define TOGGLE_TIMING() print_timings = !print_timings
#define START_FRAME() PRINT_TIMING("FRAME TIME", frame_start_time); printf("\n"); frame_start_time = GetMicroTicks()
#define BEGIN_TIMED_BLOCK(id) uint64_t _timed_block_##id = GetMicroTicks()
#define END_TIMED_BLOCK(id) PRINT_TIMING(#id, _timed_block_##id)
#define TIMED_SCOPE(id) ScopeTimer _timed_scope_##id(__FUNCTION__ " " #id)
#define TIMED_FUNCTION() ScopeTimer JOIN(_timed_function_, __LINE__)(__FUNCTION__)

#endif // TIMING_H

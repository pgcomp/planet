#ifndef PP_H
#define PP_H

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


#endif // PP_H

#ifndef LOGGING_H
#define LOGGING_H

#include <cstdio>

#define LOG_WARNING(fmt, ...) fprintf(stdout, "[WARNING] " fmt "\n", ##__VA_ARGS__)
#define LOG_ERROR(fmt, ...) fprintf(stderr, "[ERROR] " fmt "\n", ##__VA_ARGS__)

#endif // LOGGING_H

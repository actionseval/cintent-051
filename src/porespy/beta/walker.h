#pragma once
#include <stdlib.h>

#define PRINT_DB_LINE printf("%s line %d\n", __FUNCTION__, __LINE__);

#define RAND_MID (RAND_MAX / 2)

#ifndef _WIN32
#define max(a, b) \
({ __typeof__ (a) _a = (a); \
    __typeof__ (b) _b = (b); \
  _a >= _b ? _a : _b; })

  
#define min(a, b) \
({ __typeof__ (a) _a = (a); \
    __typeof__ (b) _b = (b); \
  _a <= _b ? _a : _b; })
#endif

// Edge or inside
enum BorderStatus {
    LEFT,
    RIGHT,
    TOP,
    BOTTOM,
    INSIDE,
    // For 3D images:
    FRONT,
    BACK
};

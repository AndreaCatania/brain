// *Really* minimal PCG32 code / (c) 2014 M.E. O'Neill / pcg-random.org
// Licensed under Apache License 2.0 (NO WARRANTY, etc. see website)

#ifndef RANDOM_H
#define RANDOM_H

#include "brain/typedefs.h"

#define PCG_DEFAULT_INC_64 1442695040888963407ULL

typedef struct { uint64_t state;  uint64_t inc; } pcg32_random_t;
uint32_t pcg32_random_r(pcg32_random_t* rng);

#endif // RANDOM_H

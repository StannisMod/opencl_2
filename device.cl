typedef struct {
    float aggregate;
    float inclusivePrefix;
    //int status;  // 0 - no info, 1 - aggregate available, 2 - inclusivePrefix available
    //int dummy;
} PARTITION;

kernel void aggregate(const global float* src, global float* dst, global PARTITION* p) {
    size_t ind = get_global_id(0);
    size_t part_start = ind * BATCH_SIZE;

// 1 & 2 done on host
// 3

    global PARTITION* me = &p[ind];
    me->aggregate = NAN;
    me->inclusivePrefix = NAN;

    float sum = 0;
    for (size_t i = part_start; i < part_start + BATCH_SIZE; i++) {
        sum += src[i];
    }

    me->aggregate = sum;
    if (ind == 0) {
        me->inclusivePrefix = sum;
    }
}

kernel void reduce(const global float* src, global float* dst, global PARTITION* p) {
    size_t ind = get_global_id(0);
    size_t part_start = ind * BATCH_SIZE;
// 4 & 5
    global PARTITION* me = &p[ind];

    float aggregate = 0;
    for (size_t i = ind; i >= 1; i--) {
        global PARTITION* it = &p[i - 1];

        if (it->inclusivePrefix != NAN) {
            aggregate += it->inclusivePrefix;
            me->inclusivePrefix = aggregate + me->aggregate;
            break;
        } else {
            aggregate += it->aggregate;
        }
    }

    float res = 0;
    for (size_t i = part_start; i < part_start + BATCH_SIZE; i++) {
        res += src[i];
        dst[i] = res + aggregate;
    }
}
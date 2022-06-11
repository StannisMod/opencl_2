typedef struct {
    float aggregate;
    float inclusivePrefix;
    int status;  // 0 - no info, 1 - aggregate available, 2 - inclusivePrefix available
    int dummy;
} PARTITION;

kernel void prefix(const global float* src, global float* dst, const int n, global PARTITION* p) {
    int ind = get_global_id(0);
    int part_start = ind * BATCH_SIZE;

// 1 & 2 done on host
// 3

    barrier(CLK_GLOBAL_MEM_FENCE);

    float sum = 0;
    for (int i = part_start; i < part_start + BATCH_SIZE; i++) {
        sum += src[i];
    }

    //PARTITION me = p[ind];
    p[ind].aggregate = sum;
    p[ind].status = 1;
    if (ind == 0) {
        p[ind].inclusivePrefix = sum;
        p[ind].status = 2;
    }

    //p[ind] = me;

    barrier(CLK_GLOBAL_MEM_FENCE);

// 4 & 5
    float aggregate = 0;
    for (int i = ind - 1; i >= 0; i--) {
        int status;
        // spin lock
        while ((status = p[i].status) == 0);

        if (status == 1) {
            aggregate += p[i].aggregate;
        } else {
            aggregate += p[i].inclusivePrefix;
            p[ind].inclusivePrefix = aggregate + sum;
            p[ind].status = 2;
            //p[ind] = me;
            break;
        }
    }

    barrier(CLK_GLOBAL_MEM_FENCE);

    sum = 0;
    for (int i = part_start; i < part_start + BATCH_SIZE; i++) {
        sum += src[i];
        dst[i] = sum + aggregate;
    }
}
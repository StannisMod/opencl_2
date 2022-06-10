kernel void prefix(const global float* src, global float* dst, const int n) {
    size_t ind = get_global_id(0);
    float sum = 0;

    for (int i = 0; i <= ind; i++) {
        sum += src[i];
    }

    dst[ind] = sum;
}
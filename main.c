#include <math.h>
#include <time.h>

#include "opencl_lab2.h"

#define GENERATE 0
#define CHECK 1

char* readAll(FILE *F, size_t *len) {
    fseek(F, 0, SEEK_END);
    *len = ftell(F);
    fseek(F, 0, SEEK_SET);
    char* data = malloc(*len * sizeof(char));

    fread(data, sizeof(char), *len, F);
    return data;
}

int checkErr(cl_int err, char* errorMsg) {
    if (err != CL_SUCCESS) {
        printf("%s: %i\n", errorMsg, err);
        return 1;
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 3 + 1) {
        printf("Usage: <device index> <input file> <output file>");
        return 1;
    }

    int deviceIndex;
    sscanf(argv[1], "%i", &deviceIndex);
    char* inputFile = argv[2];
    char* outputFile = argv[3];

    FILE *F;

#if GENERATE
    if (generate(100000) != 0) {
        return 1;
    }
#endif

    if ((F = fopen("../device.cl", "rb")) == NULL) {
        printf("Can't open file with device code, aborting\n");
        return 1;
    }

    size_t deviceCodeLen;
    char* deviceCode = readAll(F, &deviceCodeLen);
    fclose(F);

    cl_device_id device = getPreferredDevice(deviceIndex);
    if (device == 0) {
        free(deviceCode);
        return 1;
    }

    cl_int errCode = 0;

    // 3. Create a context and command queue on that device.
    cl_context context = clCreateContext( NULL,
                                          1,
                                          &device,
                                          NULL, NULL, &errCode);
    if (checkErr(errCode, "Error on create context")) {
        free(deviceCode);
        return 1;
    }

    cl_command_queue queue = clCreateCommandQueue(context,
                                                  device,
                                                  CL_QUEUE_PROFILING_ENABLE, &errCode);
    if (checkErr(errCode, "Error on create command queue")) {
        clReleaseContext(context);
        free(deviceCode);
        return 1;
    }

    // 4. Perform runtime source compilation, and obtain kernel entry point.
    cl_program program = clCreateProgramWithSource(context,
                                                   1,
                                                   &deviceCode,
                                                   &deviceCodeLen, &errCode);

    if (checkErr(errCode, "Error on create program")) {
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        free(deviceCode);
        return 1;
    }

    cl_ulong localMemSize;
    errCode = clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &localMemSize, 0);
    if (checkErr(errCode, "Error on get local memory size")) {
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        free(deviceCode);
        return 1;
    }

    const int batchSize = 256;

    //printf("Batch size: %i\n", batchSize);

    char options[100];
    sprintf(options, "-D BATCH_SIZE=%i", batchSize);

    if ((errCode = clBuildProgram(program, 1, &device, options, NULL, NULL)) != 0) {
        printf("Error on build program: %i\n", errCode);

        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

        char *log = malloc(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

        printf("%s\n", log);
        free(log);

        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        free(deviceCode);
        return 1;
    }

    free(deviceCode);

    cl_kernel kernel = clCreateKernel(program, "prefix", &errCode);

    if (checkErr(errCode, "Can't create kernel")) {
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return 1;
    }

    size_t localMemUsing;
    errCode = clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_LOCAL_MEM_SIZE, sizeof(size_t), &localMemUsing, 0);
    size_t workGroupSize;
    errCode = clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &workGroupSize, 0);

    //printf("Local memory used: %zu bytes, available: %zu bytes, workgroup size: %zu\n", localMemUsing, localMemSize, workGroupSize);

    if ((F = fopen(inputFile, "r")) == NULL) {
        printf("Can't open input file, aborting\n");
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return 1;
    }

    int n;
    fscanf(F, "%i", &n);

    int n_2 = ((n % batchSize) == 0) ? n : (n + batchSize - (n % batchSize));

    cl_float *buf = malloc(n_2 * sizeof (cl_float));
    if (buf == NULL) {
        printf("Not enough memory");
        fclose(F);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return 1;
    }

    memset(buf, 0, n_2 * sizeof (cl_float));
    for (int i = 0; i < n; i++) {
        fscanf(F, "%f", &buf[i]);
    }

    cl_mem srcBuffer = clCreateBuffer(context,
                                          CL_MEM_READ_ONLY,
                                          n_2 * sizeof(cl_float),
                                          NULL, &errCode);
    if (checkErr(errCode, "srcBuffer not created")) {
        free(buf);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return 1;
    }

    cl_mem resBuffer = clCreateBuffer(context,
                                      CL_MEM_READ_ONLY,
                                      n_2 * sizeof(cl_float),
                                      NULL, &errCode);
    if (checkErr(errCode, "resBuffer not created")) {
        free(buf);
        clReleaseMemObject(srcBuffer);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return 1;
    }

    struct {
        float aggregate;
        float inclusivePrefix;
        int status;
        int dummy; // for sizeof(PARTITION) == 16
    } PARTITION;

    PARTITION.aggregate = 0;
    PARTITION.inclusivePrefix = 0;
    PARTITION.status = 0;
    PARTITION.dummy = 0;

    cl_mem partBuffer = clCreateBuffer(context,
                                      CL_MEM_READ_WRITE,
                                      n_2 / batchSize * sizeof(PARTITION),
                                      NULL, &errCode);
    if (checkErr(errCode, "partBuffer not created")) {
        free(buf);
        clReleaseMemObject(srcBuffer);
        clReleaseMemObject(resBuffer);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return 1;
    }

    clSetKernelArg(kernel, 0, sizeof(srcBuffer), (void*) &srcBuffer);
    clSetKernelArg(kernel, 1, sizeof(resBuffer), (void*) &resBuffer);
    clSetKernelArg(kernel, 2, sizeof(int), &n_2);
    clSetKernelArg(kernel, 3, sizeof(partBuffer), (void*) &partBuffer);

    clock_t startTime = clock();

    errCode = clEnqueueWriteBuffer(queue,
                                   srcBuffer,
                                   CL_TRUE,
                                   0,
                                   n_2 * sizeof(cl_float),
                                   buf, 0, NULL, NULL);

    if (checkErr(errCode, "Error on enqueue (host -> device) buffer")) {
        clReleaseMemObject(srcBuffer);
        clReleaseMemObject(resBuffer);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return 1;
    }

    errCode = clEnqueueFillBuffer(queue,
                                  partBuffer,
                                  &PARTITION,
                                  sizeof(PARTITION),
                                  0,
                                  n_2 / batchSize * sizeof(PARTITION),
                                  0, NULL, NULL);

    if (checkErr(errCode, "Error on enqueue filling device buffer")) {
        clReleaseMemObject(srcBuffer);
        clReleaseMemObject(resBuffer);
        clReleaseMemObject(partBuffer);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return 1;
    }

    cl_event event;

    size_t global_work_size = n_2 / batchSize;

    clEnqueueNDRangeKernel(queue,
                           kernel,
                           1,
                           NULL,
                           &global_work_size,
                           NULL, // TODO Find local work size
                           0,
                           NULL, &event);

    clWaitForEvents(1, &event);
    clFinish( queue);

    errCode = clEnqueueReadBuffer(queue,
                                  resBuffer,
                                  CL_TRUE,
                                  0,
                                  n_2 * sizeof(cl_float),
                                  buf, 0, NULL, NULL);

    if (checkErr(errCode, "Error on enqueue (device -> host) buffer")) {
        clReleaseMemObject(srcBuffer);
        clReleaseMemObject(resBuffer);
        clReleaseMemObject(partBuffer);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return 1;
    }

    double fullTime = (double)(clock() - startTime) / CLOCKS_PER_SEC * 1000;

    cl_ulong time_start = 0;
    cl_ulong time_end = 0;

    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);

    double computeTime = (double)(time_end - time_start) / 1000000;

    printf("\nTime: %f\t%f \n", computeTime, fullTime);

    F = fopen(outputFile, "wb");
    if (F == NULL) {
        printf("Can't open output file\n");
        free(buf);
        return 1;
    }

    fprintf(F, "%i\n", n);

    for (int i = 0; i < n; i++) {
        fprintf(F, "%f ", buf[i]);
    }
    fclose(F);

#if CHECK
    F = fopen("generated_res.out", "rb");
    if (F == NULL) {
        printf("Check failed: can't open file\n");
    } else {
        int new_n;
        fscanf(F, "%i", &new_n);
        if (new_n != n) {
            printf("Dimensions are not right\n");
        } else {
            size_t errors = 0;
            for (int i = 0; i < n; i++) {
                float expected;
                fscanf(F, "%f", &expected);
                if (fabs(expected - buf[i]) > 0.000001) {
                    errors++;
                }
            }
            if (errors == 0) {
                printf("Correctness: OK\n");
            } else {
                printf("Correctness: %zu errors NOT OK\n", errors);
            }
        }
        fclose(F);
    }
#endif

    clReleaseMemObject(srcBuffer);
    clReleaseMemObject(resBuffer);
    clReleaseMemObject(partBuffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
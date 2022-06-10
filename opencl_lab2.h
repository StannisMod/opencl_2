#ifndef OPENCL_LAB1_OPENCL_LAB1_H
#define OPENCL_LAB1_OPENCL_LAB1_H

#endif //OPENCL_LAB1_OPENCL_LAB1_H

#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

cl_device_id getPreferredDevice(int index);

int generate(int n, int k, int m);

int checkErr(cl_int err, char* errorMsg);

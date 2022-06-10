#ifndef OPENCL_LAB2_OPENCL_LAB2_H
#define OPENCL_LAB2_OPENCL_LAB2_H

#endif //OPENCL_LAB2_OPENCL_LAB2_H

#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

cl_device_id getPreferredDevice(int index);

int generate(int n);

int checkErr(cl_int err, char* errorMsg);

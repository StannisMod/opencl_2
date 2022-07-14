## Array prefix sum using decoupled-lookback technic

An OpenCL 1.2 program that efficiently computes prefix sum of float array
using decoupled-lookback

#### Run: opencl_lab2 device_index input_file output_file

On start, program will sort the OpenCL devices in order:
1. Discrete GPUs
2. Integrated GPUs
3. CPUs

Other types of OpenCL devices can be placed in one of these categories. 
This device ranging allows user to run program on the most powerful device, 
or just define the device using "index of power". Order inside categories is
undefined.

Also, the project contains file `generator.c`, that presents the generator of
samples for this program.

### Input file template
```
N
N floats - input array 
```
### Output file template
```
N
N floats - output array 
```

In the end of prefix sum performing execution time will be printed. Format:
```
Time: <execution time on device> <full execution time>
```
This time not contain time for reading input and writing output.
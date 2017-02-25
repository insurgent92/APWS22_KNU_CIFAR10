#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include "cnn.h"
#include "math.h"
#include "timer.h"

cl_mem w1_1, b1_1, w1_2, b1_2, w2_1, b2_1, w2_2, b2_2, w3_1, b3_1, w3_2, b3_2, w3_3, b3_3, w4_1, b4_1, w4_2, b4_2, w4_3, b4_3, w5_1, b5_1, w5_2, b5_2, w5_3, b5_3,w1,w2,w3,b1,b2,b3;
cl_mem c1_1, c1_2, p1, c2_1, c2_2, p2, c3_1, c3_2, c3_3, p3, c4_1, c4_2, c4_3, p4, c5_1, c5_2, c5_3, p5,fc1,fc2,fc3;
cl_mem mem_image;

cl_kernel pooling_kernel, convolution_layer_kernel, convolution_bias_kernel, set_zero_kernel, convolution_first_kernel,fc_kernel, sm_kernel;
cl_command_queue queue;
cl_context context;
cl_platform_id platform;
cl_program program;
cl_device_id device;
cl_int err;
#define PARALLEL_IMAGE 3000

#define CHECK_ERROR(err) \
  if (err != CL_SUCCESS) { \
    printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
    exit(EXIT_FAILURE); \
    }
#define ReLU(x) (((x)>0)?(x):0)
static void fc_layer(cl_mem input_neuron, cl_mem output_neuron, cl_mem weights, cl_mem biases, int M, int N) {

	err = clSetKernelArg(fc_kernel, 0, sizeof(cl_mem), &input_neuron);
	CHECK_ERROR(err);
	err = clSetKernelArg(fc_kernel, 1, sizeof(cl_mem), &output_neuron);
	CHECK_ERROR(err);
	err = clSetKernelArg(fc_kernel, 2, sizeof(cl_mem), &weights);
	CHECK_ERROR(err);
	err = clSetKernelArg(fc_kernel, 3, sizeof(cl_mem), &biases);
	CHECK_ERROR(err);
	err = clSetKernelArg(fc_kernel, 4, sizeof(cl_int), &M);
	CHECK_ERROR(err);
	err = clSetKernelArg(fc_kernel, 5, sizeof(cl_int), &N);
	CHECK_ERROR(err);

	size_t global_size = PARALLEL_IMAGE;
	size_t local_size = 64;
	global_size = (global_size + local_size - 1) / local_size * local_size;

	err = clEnqueueNDRangeKernel(queue, fc_kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
	CHECK_ERROR(err);

	
}

static void init_Membuffer() {

	mem_image = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * 32 * 32 * 3 * PARALLEL_IMAGE, NULL, &err); CHECK_ERROR(err);

	w1_1 = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * 64 * 3 * 3 * 3, NULL, &err); CHECK_ERROR(err);
	b1_1 = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * 64, NULL, &err); CHECK_ERROR(err);
	w1_2 = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * 64 * 64 * 3 * 3, NULL, &err); CHECK_ERROR(err);
	b1_2 = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * 64, NULL, &err); CHECK_ERROR(err);
	w2_1 = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * 128 * 64 * 3 * 3, NULL, &err); CHECK_ERROR(err);
	b2_1 = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * 128, NULL, &err); CHECK_ERROR(err);
	w2_2 = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * 128 * 128 * 3 * 3, NULL, &err); CHECK_ERROR(err);
	b2_2 = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * 128, NULL, &err); CHECK_ERROR(err);
	w3_1 = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * 256 * 128 * 3 * 3, NULL, &err); CHECK_ERROR(err);
	b3_1 = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * 256, NULL, &err); CHECK_ERROR(err);
	w3_2 = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * 256 * 256 * 3 * 3, NULL, &err); CHECK_ERROR(err);
	b3_2 = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * 256, NULL, &err); CHECK_ERROR(err);
	w3_3 = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * 256 * 256 * 3 * 3, NULL, &err); CHECK_ERROR(err);
	b3_3 = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * 256, NULL, &err); CHECK_ERROR(err);
	w4_1 = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * 512 * 256 * 3 * 3, NULL, &err); CHECK_ERROR(err);
	b4_1 = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * 512, NULL, &err); CHECK_ERROR(err);
	w4_2 = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * 512 * 512 * 3 * 3, NULL, &err); CHECK_ERROR(err);
	b4_2 = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * 512, NULL, &err); CHECK_ERROR(err);
	w4_3 = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * 512 * 512 * 3 * 3, NULL, &err); CHECK_ERROR(err);
	b4_3 = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * 512, NULL, &err); CHECK_ERROR(err);
	w5_1 = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * 512 * 512 * 3 * 3, NULL, &err); CHECK_ERROR(err);
	b5_1 = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * 512, NULL, &err); CHECK_ERROR(err);
	w5_2 = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * 512 * 512 * 3 * 3, NULL, &err); CHECK_ERROR(err);
	b5_2 = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * 512, NULL, &err); CHECK_ERROR(err);
	w5_3 = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * 512 * 512 * 3 * 3, NULL, &err); CHECK_ERROR(err);
	b5_3 = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * 512, NULL, &err); CHECK_ERROR(err);
	w1 = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * 512 * 512 , NULL, &err); CHECK_ERROR(err);
	b1 = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * 512, NULL, &err); CHECK_ERROR(err);
	w2 = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * 512 * 512, NULL, &err); CHECK_ERROR(err);
	b2 = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * 512, NULL, &err); CHECK_ERROR(err);
	w3 = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * 10 * 512 , NULL, &err); CHECK_ERROR(err);
	b3 = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * 10, NULL, &err); CHECK_ERROR(err);


	c1_1 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 64 * 32 * 32 * PARALLEL_IMAGE, NULL, &err); CHECK_ERROR(err);
	c1_2 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 64 * 32 * 32 * PARALLEL_IMAGE, NULL, &err); CHECK_ERROR(err);

	p1 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 64 * 16 * 16 * PARALLEL_IMAGE, NULL, &err); CHECK_ERROR(err);
	c2_1 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 128 * 16 * 16 * PARALLEL_IMAGE, NULL, &err); CHECK_ERROR(err);
	c2_2 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 128 * 16 * 16 * PARALLEL_IMAGE, NULL, &err); CHECK_ERROR(err);
	p2 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 128 * 16 * 16 * PARALLEL_IMAGE, NULL, &err); CHECK_ERROR(err);
	c3_1 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 256 * 8 * 8 * PARALLEL_IMAGE, NULL, &err); CHECK_ERROR(err);
	c3_2 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 256 * 8 * 8 * PARALLEL_IMAGE, NULL, &err); CHECK_ERROR(err);
	c3_3 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 256 * 8 * 8 * PARALLEL_IMAGE, NULL, &err); CHECK_ERROR(err);
	p3 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 256 * 4 * 4 * PARALLEL_IMAGE, NULL, &err); CHECK_ERROR(err);
	c4_1 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 512 * 4 * 4 * PARALLEL_IMAGE, NULL, &err); CHECK_ERROR(err);
	c4_2 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 512 * 4 * 4 * PARALLEL_IMAGE, NULL, &err); CHECK_ERROR(err);
	c4_3 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 512 * 4 * 4 * PARALLEL_IMAGE, NULL, &err); CHECK_ERROR(err);
	p4 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 512 * 2 * 2 * PARALLEL_IMAGE, NULL, &err); CHECK_ERROR(err);
	c5_1 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 512 * 2 * 2 * PARALLEL_IMAGE, NULL, &err); CHECK_ERROR(err);
	c5_2 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 512 * 2 * 2 * PARALLEL_IMAGE, NULL, &err); CHECK_ERROR(err);
	c5_3 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 512 * 2 * 2 * PARALLEL_IMAGE, NULL, &err); CHECK_ERROR(err);
	p5 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 512 * 1 * 1 * PARALLEL_IMAGE, NULL, &err); CHECK_ERROR(err);

	fc1 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 512   * PARALLEL_IMAGE, NULL, &err); CHECK_ERROR(err);
	fc2 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 512  * PARALLEL_IMAGE, NULL, &err); CHECK_ERROR(err);
	fc3 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 10  * PARALLEL_IMAGE, NULL, &err); CHECK_ERROR(err);
}

char* get_source_code(const char*file_name, size_t *len) {
	char *source_code = NULL;
	size_t source_length = 0;
	FILE *file = fopen(file_name, "r");

	if (file == NULL) {
		printf("[%s:%d] Fariled to open %s\n", __FILE__, __LINE__, file_name);
		exit(EXIT_FAILURE);
	}

	fseek(file, 0, SEEK_END);
	source_length = (size_t)ftell(file);
	rewind(file);
	source_code = (char*)malloc(source_length + 1);
	fread(source_code, source_length, 1, file);
	source_code[source_length] = '\0';
	fclose(file);
	*len = source_length;
	return source_code;
}

static void init_Opencl() {
	char* kernel_source = NULL;
	size_t kernel_source_size = 0;

	err = clGetPlatformIDs(1, &platform, NULL);
	CHECK_ERROR(err);

	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
	CHECK_ERROR(err);

	context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
	CHECK_ERROR(err);

	queue = clCreateCommandQueue(context, device, 0, &err);
	CHECK_ERROR(err);

	kernel_source = get_source_code("kernel.cl", &kernel_source_size);

	program = clCreateProgramWithSource(context, 1, (const char**)&kernel_source, &kernel_source_size, &err);
	CHECK_ERROR(err);

	err = clBuildProgram(program, 1, &device, "-cl-fast-relaxed-math -w -cl-unsafe-math-optimizations", NULL, NULL);

	if (err == CL_BUILD_PROGRAM_FAILURE) {
		size_t log_size;
		char* log;

		err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
		CHECK_ERROR(err);

		log = (char*)malloc(log_size + 1);

		err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
		CHECK_ERROR(err);

		log[log_size] = '\0';
		printf("Compiler error:\n%s\n", log);

		free(log);

		exit(0);
	}

	CHECK_ERROR(err);

	//커널 초기화
	pooling_kernel = clCreateKernel(program, "pooling_layer", &err);
	CHECK_ERROR(err);
	
	convolution_first_kernel = clCreateKernel(program, "convolution_first", &err);
	CHECK_ERROR(err);
	convolution_bias_kernel = clCreateKernel(program, "convolution_bias", &err);
	CHECK_ERROR(err);
	fc_kernel = clCreateKernel(program, "fully_connect", &err);
	CHECK_ERROR(err);
	sm_kernel = clCreateKernel(program, "soft_max", &err);
	CHECK_ERROR(err);
}

static void pooling_layer(cl_mem inputs, cl_mem outputs, int D, int N) {

	(clSetKernelArg(pooling_kernel, 0, sizeof(cl_mem), &inputs));
	(clSetKernelArg(pooling_kernel, 1, sizeof(cl_mem), &outputs));
	(clSetKernelArg(pooling_kernel, 2, sizeof(int), &D));
	(clSetKernelArg(pooling_kernel, 3, sizeof(int), &N));

	size_t global_size[3] = { N*N, D, PARALLEL_IMAGE };
	size_t local_size[3] = { 8, 8, 4 };

	global_size[0] = (global_size[0] + local_size[0] - 1) / local_size[0] * local_size[0];
	global_size[1] = (global_size[1] + local_size[1] - 1) / local_size[1] * local_size[1];
	global_size[2] = (global_size[2] + local_size[2] - 1) / local_size[2] * local_size[2];

	(clEnqueueNDRangeKernel(queue, pooling_kernel, 3, NULL, global_size, local_size, 0, NULL, NULL));
}

static void writeBuffers(float **network) {
	(clEnqueueWriteBuffer(queue, w1_1, CL_FALSE, 0, sizeof(float) * 64 * 3 * 3 * 3, network[0], 0, NULL, NULL));
	(clEnqueueWriteBuffer(queue, b1_1, CL_FALSE, 0, sizeof(float) * 64, network[1], 0, NULL, NULL));
	(clEnqueueWriteBuffer(queue, w1_2, CL_FALSE, 0, sizeof(float) * 64 * 64 * 3 * 3, network[2], 0, NULL, NULL));
	(clEnqueueWriteBuffer(queue, b1_2, CL_FALSE, 0, sizeof(float) * 64, network[3], 0, NULL, NULL));
	(clEnqueueWriteBuffer(queue, w2_1, CL_FALSE, 0, sizeof(float) * 128 * 64 * 3 * 3, network[4], 0, NULL, NULL));
	(clEnqueueWriteBuffer(queue, b2_1, CL_FALSE, 0, sizeof(float) * 128, network[5], 0, NULL, NULL));
	(clEnqueueWriteBuffer(queue, w2_2, CL_FALSE, 0, sizeof(float) * 128 * 128 * 3 * 3, network[6], 0, NULL, NULL));
	(clEnqueueWriteBuffer(queue, b2_2, CL_FALSE, 0, sizeof(float) * 128, network[7], 0, NULL, NULL));
	(clEnqueueWriteBuffer(queue, w3_1, CL_FALSE, 0, sizeof(float) * 256 * 128 * 3 * 3, network[8], 0, NULL, NULL));
	(clEnqueueWriteBuffer(queue, b3_1, CL_FALSE, 0, sizeof(float) * 256, network[9], 0, NULL, NULL));
	(clEnqueueWriteBuffer(queue, w3_2, CL_FALSE, 0, sizeof(float) * 256 * 256 * 3 * 3, network[10], 0, NULL, NULL));
	(clEnqueueWriteBuffer(queue, b3_2, CL_FALSE, 0, sizeof(float) * 256, network[11], 0, NULL, NULL));
	(clEnqueueWriteBuffer(queue, w3_3, CL_FALSE, 0, sizeof(float) * 256 * 256 * 3 * 3, network[12], 0, NULL, NULL));
	(clEnqueueWriteBuffer(queue, b3_3, CL_FALSE, 0, sizeof(float) * 256, network[13], 0, NULL, NULL));
	(clEnqueueWriteBuffer(queue, w4_1, CL_FALSE, 0, sizeof(float) * 512 * 256 * 3 * 3, network[14], 0, NULL, NULL));
	(clEnqueueWriteBuffer(queue, b4_1, CL_FALSE, 0, sizeof(float) * 512, network[15], 0, NULL, NULL));
	(clEnqueueWriteBuffer(queue, w4_2, CL_FALSE, 0, sizeof(float) * 512 * 512 * 3 * 3, network[16], 0, NULL, NULL));
	(clEnqueueWriteBuffer(queue, b4_2, CL_FALSE, 0, sizeof(float) * 512, network[17], 0, NULL, NULL));
	(clEnqueueWriteBuffer(queue, w4_3, CL_FALSE, 0, sizeof(float) * 512 * 512 * 3 * 3, network[18], 0, NULL, NULL));
	(clEnqueueWriteBuffer(queue, b4_3, CL_FALSE, 0, sizeof(float) * 512, network[19], 0, NULL, NULL));
	(clEnqueueWriteBuffer(queue, w5_1, CL_FALSE, 0, sizeof(float) * 512 * 512 * 3 * 3, network[20], 0, NULL, NULL));
	(clEnqueueWriteBuffer(queue, b5_1, CL_FALSE, 0, sizeof(float) * 512, network[21], 0, NULL, NULL));
	(clEnqueueWriteBuffer(queue, w5_2, CL_FALSE, 0, sizeof(float) * 512 * 512 * 3 * 3, network[22], 0, NULL, NULL));
	(clEnqueueWriteBuffer(queue, b5_2, CL_FALSE, 0, sizeof(float) * 512, network[23], 0, NULL, NULL));
	(clEnqueueWriteBuffer(queue, w5_3, CL_FALSE, 0, sizeof(float) * 512 * 512 * 3 * 3, network[24], 0, NULL, NULL));
	(clEnqueueWriteBuffer(queue, b5_3, CL_FALSE, 0, sizeof(float) * 512, network[25], 0, NULL, NULL));

	(clEnqueueWriteBuffer(queue, w1, CL_FALSE, 0, sizeof(float) * 512 * 512 , network[26], 0, NULL, NULL));
	(clEnqueueWriteBuffer(queue, b1, CL_FALSE, 0, sizeof(float) * 512, network[27], 0, NULL, NULL));
	(clEnqueueWriteBuffer(queue, w2, CL_FALSE, 0, sizeof(float) * 512 * 512 , network[28], 0, NULL, NULL));
	(clEnqueueWriteBuffer(queue, b2, CL_FALSE, 0, sizeof(float) * 512, network[29], 0, NULL, NULL));
	(clEnqueueWriteBuffer(queue, w3, CL_FALSE, 0, sizeof(float) * 512 * 10, network[30], 0, NULL, NULL));
	(clEnqueueWriteBuffer(queue, b3, CL_FALSE, 0, sizeof(float) * 10, network[31], 0, NULL, NULL));
}

static void convolution_layer(cl_mem inputs, cl_mem outputs, cl_mem filters, cl_mem biases, int D2, int D1, int N) {
	
	err = clSetKernelArg(convolution_first_kernel, 0, sizeof(cl_mem), &inputs);
	CHECK_ERROR(err);
	err = clSetKernelArg(convolution_first_kernel, 1, sizeof(cl_mem), &outputs);
	CHECK_ERROR(err);
	err = clSetKernelArg(convolution_first_kernel, 2, sizeof(cl_mem), &filters);
	CHECK_ERROR(err);
	err = clSetKernelArg(convolution_first_kernel, 3, sizeof(cl_mem), &biases);
	CHECK_ERROR(err);
	err = clSetKernelArg(convolution_first_kernel, 4, sizeof(cl_int), &D2);
	CHECK_ERROR(err);
	err = clSetKernelArg(convolution_first_kernel, 5, sizeof(cl_int), &D1);
	CHECK_ERROR(err);
	err = clSetKernelArg(convolution_first_kernel, 6, sizeof(cl_int), &N);
	CHECK_ERROR(err);
	
	size_t global_size[3] = { N*N ,D2,PARALLEL_IMAGE /8};
	size_t local_size[3] = { 1,8 ,8  };
	if (N*N >= 256) {
		local_size[0] = 256;
		local_size[1] = 1;
		local_size[2] = 1;
	}
	else if (N*N >= 64) {

		local_size[0] = 64;
		local_size[1] = 4;
		local_size[2] = 1;
	}

	else if (N*N >= 16) {
		local_size[0] = 16;
		local_size[1] = 4;
		local_size[2] = 1;
	}
	else  {
		local_size[0] = 1;
		local_size[1] = 32;
		local_size[2] = 8;
	}
	
	global_size[0] = (global_size[0] + local_size[0] - 1) / local_size[0] * local_size[0];
	global_size[1] = (global_size[1] + local_size[1] - 1) / local_size[1] * local_size[1];
	global_size[2] = (global_size[2] + local_size[2] - 1) / local_size[2] * local_size[2];
	err = clEnqueueNDRangeKernel(queue, convolution_first_kernel, 3, NULL, global_size, local_size, 0, NULL, NULL);
	CHECK_ERROR(err);

}
void cnnfree() {}
void cnn_init() {

	init_Opencl();
	init_Membuffer();
}


float* alloc_layer(size_t n) {
	return (float*)malloc(n * sizeof(float));
}

static void softmax(cl_mem output, int N) {

	err = clSetKernelArg(sm_kernel, 0, sizeof(cl_mem), &output);
	CHECK_ERROR(err);
	err = clSetKernelArg(sm_kernel, 1, sizeof(cl_int), &N);
	CHECK_ERROR(err);

	size_t global_size = PARALLEL_IMAGE;
	size_t local_size = 64;
	global_size = (global_size + local_size - 1) / local_size * local_size;

	err = clEnqueueNDRangeKernel(queue, sm_kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
	CHECK_ERROR(err);
	
}

static int find_max(float *fc, int N) {
	int i;
	int maxid = 0;
	float maxval = 0;
	for (i = 0; i < N; i++) {
		if (maxval < fc[i]) {
			maxval = fc[i];
			maxid = i;
		}
	}
	return maxid;
}



void cnn(float *images, float **network, int *labels, float *confidences, int num_images) {
	timer_start(1);
	writeBuffers(network);
	clFinish(queue);
	printf("WriteBuffer..%f\n", timer_end(1));

	
	// run network
	for (int i = 0; i < num_images; i += PARALLEL_IMAGE)
	{

		float *image = images + i * 3 * 32 * 32;
		(clEnqueueWriteBuffer(queue, mem_image, CL_FALSE, 0, sizeof(float) * 32 * 32 * 3 * PARALLEL_IMAGE, image, 0, NULL, NULL));
		timer_start(1);
		convolution_layer(mem_image, c1_1, w1_1, b1_1, 64, 3, 32);
		clFinish(queue);
		printf("c1_1..%f\n", timer_end(1));

		timer_start(1);
		convolution_layer(c1_1, c1_2, w1_2, b1_2, 64, 64, 32);
		clFinish(queue);
		printf("c1_2..%f\n", timer_end(1));

		timer_start(1);
		pooling_layer(c1_2, p1, 64, 16);
		clFinish(queue);
		printf("p1..%f\n", timer_end(1));

		timer_start(1);

		convolution_layer(p1, c2_1, w2_1, b2_1, 128, 64, 16);
		clFinish(queue);
		printf("c2_1..%f\n", timer_end(1));

		timer_start(1);
		convolution_layer(c2_1, c2_2, w2_2, b2_2, 128, 128, 16);
		clFinish(queue);
		printf("c2_2..%f\n", timer_end(1));

		timer_start(1);
		pooling_layer(c2_2, p2, 128, 8);
		clFinish(queue);
		printf("p2..%f\n", timer_end(1));

		timer_start(1);

		convolution_layer(p2, c3_1, w3_1, b3_1, 256, 128, 8);
		clFinish(queue);
		printf("c3_1..%f\n", timer_end(1));

		timer_start(1);
		convolution_layer(c3_1, c3_2, w3_2, b3_2, 256, 256, 8);
		clFinish(queue);
		printf("c3_2..%f\n", timer_end(1));

		timer_start(1);
		convolution_layer(c3_2, c3_3, w3_3, b3_3, 256, 256, 8);
		clFinish(queue);
		printf("c3_3..%f\n", timer_end(1));

		timer_start(1);
		pooling_layer(c3_3, p3, 256, 4);
		clFinish(queue);
		printf("p3..%f\n", timer_end(1));

		timer_start(1);

		convolution_layer(p3, c4_1, w4_1, b4_1, 512, 256, 4);
		clFinish(queue);
		printf("c4_1..%f\n", timer_end(1));

		timer_start(1);
		convolution_layer(c4_1, c4_2, w4_2, b4_2, 512, 512, 4);
		clFinish(queue);
		printf("c4_2..%f\n", timer_end(1));

		timer_start(1);
		convolution_layer(c4_2, c4_3, w4_3, b4_3, 512, 512, 4);
		clFinish(queue);
		printf("c4_3..%f\n", timer_end(1));

		timer_start(1);
		pooling_layer(c4_3, p4, 512, 2);
		clFinish(queue);
		printf("p4..%f\n", timer_end(1));

		timer_start(1);

		convolution_layer(p4, c5_1, w5_1, b5_1, 512, 512, 2);
		clFinish(queue);
		printf("c5_1..%f\n", timer_end(1));

		timer_start(1);
		convolution_layer(c5_1, c5_2, w5_2, b5_2, 512, 512, 2);
		clFinish(queue);
		printf("c5_2..%f\n", timer_end(1));

		timer_start(1);
		convolution_layer(c5_2, c5_3, w5_3, b5_3, 512, 512, 2);
		clFinish(queue);
		printf("c5_3..%f\n", timer_end(1));

		timer_start(1);
		pooling_layer(c5_3, p5, 512, 1);
		clFinish(queue);
		printf("p5..%f\n", timer_end(1));

		
		fc_layer(p5, fc1, w1, b1, 512, 512);

		fc_layer(fc1, fc2, w2, b2, 512, 512);
		fc_layer(fc2, fc3, w3, b3, 10, 512);

		softmax(fc3, 10);

	}

	float* result = (float*)malloc(sizeof(float)*10*PARALLEL_IMAGE);

	err = clEnqueueReadBuffer(queue, fc3, CL_TRUE, 0, sizeof(float) * 10 * PARALLEL_IMAGE, result, 0, NULL, NULL);
	CHECK_ERROR(err);

	
	for (int i = 0; i < num_images; i += 1)
	{
		labels[i ] = find_max(result+10*i, 10);
		confidences[i ] = result[i*10+labels[i]];
	}
		
}

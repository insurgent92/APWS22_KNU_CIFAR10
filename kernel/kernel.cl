#define PARALLEL_IMAGE 3000

#define ReLU(x) (((x)>0)?(x):0)
__kernel void fully_connect(__global float *input_neuron,__global float *output_neuron,__global float *weights,__global float *biases, int M, int N){
	
	int image_num = get_global_id(0);
	int i,j;

	if(image_num<PARALLEL_IMAGE){
		for (j = 0; j < M; j++) {
			float sum = 0.0f;
			for (i = 0; i < N; i++) {
				sum += input_neuron[N*image_num+i] * weights[j * N + i];
			}
			sum += biases[j];
			output_neuron[M*image_num+j] = ReLU(sum);
		}
	}


}
__kernel void soft_max(__global float *output,int N){
	int image_num = get_global_id(0);
	if(image_num<PARALLEL_IMAGE){
		int i;
		float max = output[image_num*N+0];
		for (i = 1; i < N; i++) {
			max = (output[image_num*N+i] > max) ? output[image_num*N+i] : max;
		}
		float sum = 0;
		for (i = 0; i < N; i++) {
			sum += exp(output[image_num*N+i] - max);
		}
		for (i = 0; i < N; i++) {
			output[image_num*N+i] = exp(output[image_num*N+i] - max) / sum;
		}
	}
}

__kernel void pooling_layer(__global float *inputs, __global float *outputs, int D, int N) {
	int idx = get_global_id(1); // 
	int ij = get_global_id(0); //i
	int i = ij % N;
	int j= ij / N;
	int p=get_global_id(2);

	if(idx<D && i<N && j<N && p<PARALLEL_IMAGE){
		__global float * input = inputs +p*N*N*4*D+ idx * N * N * 4;
		__global float * output = outputs +p*N*N*D+ idx * N * N;
	
		float max = 0;

		float pixel = input[(i * 2 ) * 2 * N + j * 2];
		max = (max > pixel) ? max : pixel;
		pixel = input[(i * 2 ) * 2 * N + j * 2 + 1];
		max = (max > pixel) ? max : pixel;
		pixel = input[(i * 2 + 1) * 2 * N + j * 2 ];
		   max = (max > pixel) ? max : pixel;
		pixel = input[(i * 2 + 1) * 2 * N + j * 2 + 1];
		max = (max > pixel) ? max : pixel;
		
		output[i * N + j] = max;
	}
}

__kernel void convolution_first(__global float *inputs,
      __global float *outputs,
      __global float *filters,
      __global float *biases,
      int outChanel, int inputChanel, int N) {
   int i = get_global_id(1);
   int j = get_global_id(0);

   int k = j % N;
   j = j / N;
   
   int img_id = get_global_id(2);

   int o,p,q;

   
   if(i < outChanel && j < N && img_id*8 < PARALLEL_IMAGE)
   {
	  float8 sum=(float8)(0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f);
	  for(q = 0; q < inputChanel; q++)
	  {
		 for(o = 0; o < 3; o++)
		 {
			for(p = 0; p < 3; p++)
			{
			   if(j + o -1 >= 0 && k + p -1 >= 0 && j + o -1 < N && k + p -1 < N) 
			   {
				  float8 inputVec= (float8)(inputs[img_id*8 * N*N*inputChanel + N*N*q+ (j + o -1) * N + (k + p -1)],
									inputs[(img_id*8+1) * N*N*inputChanel + N*N*q+ (j + o -1) * N + (k + p -1)],
									inputs[(img_id*8+2) * N*N*inputChanel + N*N*q+ (j + o -1) * N + (k + p -1)],
									inputs[(img_id*8+3) * N*N*inputChanel + N*N*q+ (j + o -1) * N + (k + p -1)],
									inputs[(img_id*8+4) * N*N*inputChanel + N*N*q+ (j + o -1) * N + (k + p -1)],
									inputs[(img_id*8+5) * N*N*inputChanel + N*N*q+ (j + o -1) * N + (k + p -1)],
									inputs[(img_id*8+6) * N*N*inputChanel + N*N*q+ (j + o -1) * N + (k + p -1)],
									inputs[(img_id*8+7) * N*N*inputChanel + N*N*q+ (j + o -1) * N + (k + p -1)]);

				  float filter = filters[3 * 3 * (i * inputChanel + q)+(3*o + p)];
				  sum +=  inputVec * filter;
			   }
			}
		 }
	  }
	  sum+= biases[i];

	  
	  outputs[img_id*8*outChanel*N*N + i*N*N+j*N + k] = ReLU(sum.s0);
	  outputs[(img_id*8+1)*outChanel*N*N + i*N*N+j*N + k] = ReLU(sum.s1);
      outputs[(img_id*8+2)*outChanel*N*N + i*N*N+j*N + k] = ReLU(sum.s2);
      outputs[(img_id*8+3)*outChanel*N*N + i*N*N+j*N + k] = ReLU(sum.s3);
	  outputs[(img_id*8+4)*outChanel*N*N + i*N*N+j*N + k] = ReLU(sum.s4);
	  outputs[(img_id*8+5)*outChanel*N*N + i*N*N+j*N + k] = ReLU(sum.s5);
      outputs[(img_id*8+6)*outChanel*N*N + i*N*N+j*N + k] = ReLU(sum.s6);
      outputs[(img_id*8+7)*outChanel*N*N + i*N*N+j*N + k] = ReLU(sum.s7);

   }
   
   
   
}
__kernel void convolution_layer(__global float *inputs, __global float* outputs, __global float* filters, int D2, int D1, int N){
	
	
	int b_group = get_group_id(0)*get_local_size(0);
	int b_local = get_local_id(0);
	int ab = b_group + b_local;
	int b=ab/N;
	int a= ab%N;

	int j_group = get_group_id(1)*get_local_size(1);
	int j_local = get_local_id(1);
	int j = j_group + j_local;

	int a_group = get_group_id(2)*get_local_size(2);
	int a_local = get_local_id(2);
	int p = a_group + a_local;
	
	int k, l, i;
	if(a<N && b<N && p<PARALLEL_IMAGE){
		__global float *output = outputs +p*N*N*D2+ N*N*j;
		for (i = 0; i < D1; i++) {
			__global float *input = inputs + p*N*N*D1+ N*N*i;
			__global float *filter = filters + 3 * 3 * (j * D1 + i);

			float sum = 0;
				
			if (a >= 1 && a < N + 1							&& b >= 1 && b < N + 1)//k 0 l 0
			sum += input[(a - 1)*N + (b - 1)] * filter[0];
			if (a >= 1 && a < N + 1							&& b + 1 >= 1 && b + 1 < N + 1)//k 0 l 1
			sum += input[(a - 1)*N + (b    )] * filter[1];
			if (a >= 1 && a < N + 1							&& b + 2 >= 1 && b + 2 < N + 1)//k 0 l 2
			sum += input[(a - 1)*N + (b + 1)] * filter[2];
			if (a + 1 >= 1 && a + 1 < N + 1					&& b >= 1 && b < N + 1 )//k 1 l 0
			sum += input[(a    )*N + (b - 1)] * filter[3];
			if (a + 1 >= 1 && a + 1 < N + 1					&& b + 1 >= 1 && b + 1 < N + 1)//k 1 l 1
			sum += input[(a    )*N + (b    )] * filter[4];
			if (a + 1 >= 1 && a + 1 < N + 1					&& b + 2 >= 1 && b + 2 < N + 1)//k 1 l 2
			sum += input[(a    )*N + (b + 1)] * filter[5];
			if (a + 2 >= 1 && a + 2 < N + 1					&& b >= 1 && b + 0 < N + 1)//k 2 l 0
			sum += input[(a + 1)*N + (b - 1)] * filter[6];
			if (a + 2 >= 1 && a + 2 < N + 1					&& b + 1 >= 1 && b + 1 < N + 1)//k 2 l 1
			sum += input[(a + 1)*N + (b    )] * filter[7];
			if (a + 2 >= 1 && a + 2 < N + 1					&& b + 2 >= 1 && b + 2 < N + 1)//k 2 l 2
			sum += input[(a + 1)*N + (b + 1)] * filter[8];

			output[a*N + b] += sum;
		}
	}
}

__kernel void convolution_bias(__global float *outputs, __global float* biases, int D2, int N){
	//int i = get_global_id(0);
	//int j = get_global_id(1);
	int i_group = get_group_id(1)*get_local_size(1);
	int i_local = get_local_id(1);
	int i = i_group + i_local;

	int j_group = get_group_id(0)*get_local_size(0);
	int j_local = get_local_id(0);
	int j = j_group + j_local;

	int p = get_global_id(2);
	if(i<D2 && j<N*N && p<PARALLEL_IMAGE)
		*(outputs +p*N*N*D2+ N*N*i + j) = ReLU(*(outputs +p*N*N*D2+ N*N*i + j) + biases[i]);
}
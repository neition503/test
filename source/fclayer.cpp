#include <stdlib.h>
#include "fclayer.h"

FC_layer::FC_layer(int input_num, int output_num) {
	in_num = input_num;
	out_num = output_num;
	w = (float *)malloc(sizeof(float)*input_num*output_num);
	b = (float *)malloc(sizeof(float)*output_num);
	grad_w = (float *)malloc(sizeof(float)*input_num*output_num);
	grad_b = (float *)malloc(sizeof(float)*output_num);
}

float* FC_layer::Forward(float *input_tensor) {
	x = input_tensor;
	float *out = (float *)malloc(sizeof(float)*out_num);

	// Out = W*X + b
	for (int i = 0; i < out_num; i++) {
		out[i] = 0;
		for (int j = 0; j < in_num; j++) {
			out[i] += x[j] * w[in_num*i + j];
		}
		out[i] += b[i];
	}

	return out;
}

float* FC_layer::Backward(float *out_grad) {
	float *x_grad = (float *)malloc(sizeof(float)*in_num);

	// dloss/dw = (dout/dw) * (dloss/dout) = x * (out_grad)
	for (int i = 0; i < out_num*in_num; i++) {
		grad_w[i] = x[i] * out_grad[i / in_num];
	}

	// dloss/dx = (dout/dx) * (dloss/dout) = sum(w * dloss/dout)
	for (int i = 0; i < in_num; i++) {
		x_grad[i] = 0;
		for (int j = 0; j < out_num; j++) {
			x_grad[i] += w[i + out_num*j] * out_grad[j];
		}
	}

	// dloss/db = (dout/db) * (dloss/dout) = 1 * (dloss/dout) = out_grad
	grad_b = out_grad;
	return x_grad;
}

FC_layer::~FC_layer() {
	free(x);
	free(w);
	free(b);
	free(grad_w);
	free(grad_b);
}

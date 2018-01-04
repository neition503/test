#include <stdlib.h>
#include <math.h>
#include "softmax.h"

Softmax::Softmax() {
}

float Softmax::Forward(float *input_tensor, int *label) {
	int class_num = sizeof(input_tensor) / sizeof(float);
	y = (float *)malloc(sizeof(float)*class_num);
	t = label;
	int predict = 0;
	float exp_sum = 0;
	float loss;

	// sum(exp(input))
	for (int i = 0; i < class_num; i++) {
		exp_sum += exp(input_tensor[i]);
	}

	// y = exp(input)/sum(exp(input))
	for (int i = 0; i < class_num; i++) {
		y[i] = exp(input_tensor[i])/exp_sum;
	}

	for (int i = 0; i < class_num; i++) {
		if (label[i] == 1) {
			predict = i;
			break;
		}
	}

	loss = -log(exp(input_tensor[predict]) / exp_sum);
	return loss;
}

float* Softmax::Backward() {
	float *x_grad = (float *)malloc(sizeof(float)*class_num);
	for (int i = 0; i < class_num; i++) {
		x_grad[i] = y[i] - t[i];
	}
	return x_grad;
}

Softmax::~Softmax() {
	free(y);
	free(t);
}
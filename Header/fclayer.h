class FC_layer {
private:
	int in_num;         // shape of input(x)
	int out_num;        // shape of output(out)
	float *x;			// input x
	float *w;		    // weight w
	float *b;           // bias b
	float *grad_w;	    // dloss/dw
	float *grad_b;		// dloss/db
public:
	FC_layer(int input_num, int output_num);	// Layer �����Լ�
	float* Forward(float* input_tensor);		// Forwarding
	float* Backward(float* out_grad);			// Backprop
	~FC_layer();								// �Ҹ���
};

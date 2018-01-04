class Softmax {
private:
	int class_num;         // # of classes
	float *y;			   // Softmax output y
	int *t;		           // Label t
public:
	Softmax();	// Layer 생성함수
	float Forward(float *input_tensor, int *label);		// Forwarding
	float* Backward();			// Backprop
	~Softmax();								// 소멸자
};

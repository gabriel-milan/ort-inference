all:
	g++ -o main src/ort_inference.cxx main.cxx -I include/ -lonnxruntime
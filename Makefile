ringer:
	mkdir -p build
	g++ -o build/ringer src/ort_inference.cxx ringer.cxx -I include/ -lonnxruntime
vgg16:
	mkdir -p build
	g++ -o build/vgg16 src/ort_inference.cxx vgg16.cxx -I include/ -lonnxruntime
run_ringer:
	./build/ringer
run_vgg16:
	./build/vgg16
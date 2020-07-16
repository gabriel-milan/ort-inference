#include <ort_inference.h>

int main (int argc, char **argv) {
  ORTInference my = ORTInference();
  my.setModelPath("model.onnx");
  my.initialize();
  my.execute();
  my.finalize();
  return 0;
}

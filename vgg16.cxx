/*
 *  Author: Gabriel Gazola Milan
 *  Date:   July 15, 2020
 *  Licensed under the MIT license
 */

#include <ort_inference.h>

/*
 *  This example uses a VGG16 model in ONNX
 *  format for prediction.
 */
int main (int argc, char **argv) {

  // --> Constructing object
  // You can also construct using:
  // ORTInference (const char *modelPath);
  ORTInference model = ORTInference();

  // --> Configuring model
  // The following getters and setters are
  // available:
  // - Environment name:
  // char *getEnvName ()
  // void setEnvName (const char *name)
  // - Batch size (not in use)
  // int64_t getBatchSize ()
  // void setBatchSize (int64_t size)
  // - Path to the ONNX file:
  // char *getModelPath ()
  // void setModelPath (const char *path)
  // - ORT Logging level:
  // OrtLoggingLevel getLoggingLevel ()
  // void setLoggingLevel (OrtLoggingLevel level)
  // - ORT optimization settings:
  // void setGraphOptimizationLevel (GraphOptimizationLevel level)
  // *** For now, I was lazy about making a new setter for
  // disabling verbose for the class, feel free to open a PR
  // on that.
  model.setModelPath("vgg16.onnx");
  model.setGraphOptimizationLevel(
    GraphOptimizationLevel::ORT_ENABLE_EXTENDED
  );

  // --> Initializing model
  // Will crete environment and session
  model.initialize();

  // --> Generating dummy data for prediction
  // Input size for VGG16 sample is 224x224x3
  size_t input_tensor_size = 224*224*3;
  std::vector<float> input_tensor_values(input_tensor_size);
  for (unsigned int i = 0; i < input_tensor_size; i++)
    input_tensor_values[i] = (float)i / (input_tensor_size + 1);

  // --> Predicting with dummy imput
  // I'll predict 10 times in order to show
  // how you can predict multiple samples
  // *** I'll show score for the first class only,
  // but VGG16 model has output shape of (1, 1000)
  for (int i = 0; i < 10; i++) {
    printf(" - Iteration #%d\n", i);
    float *floatarr = model.predict(input_tensor_values);
    printf("   . Score: %f\n", floatarr[0]);
  }

  // Finalize stuff
  model.finalize();
  return 0;
}

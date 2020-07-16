/*
 *  Author: Gabriel Gazola Milan
 *  Date:   July 15, 2020
 *  Licensed under the MIT license
 */

#include <ort_inference.h>

/*
 *  Constructors
 */
ORTInference::ORTInference() {
  sessionOptions.SetGraphOptimizationLevel(DEFAULT_OPTIMIZATION_LEVEL);
}

ORTInference::ORTInference(const char *path) {
  sessionOptions.SetGraphOptimizationLevel(DEFAULT_OPTIMIZATION_LEVEL);
  modelPath = (char *)path;
}

/*
 *  Logging
 */
void ORTInference::log (const char *message, const char *end="\n") {
  if (console) printf ("%s%s", message, end);
}

/*
 *  Initialization
 */
bool ORTInference::initialize () {

  // Environment
  log ("--> Initializing environment...", "");
  env = std::make_unique< Ort::Env > (
    loggingLevel,
    envName
  );
  log (" [OK]");

  // Session
  log ("--> Initializing session...", "");
  if ((modelPath == NULL) || (modelPath[0] == '\0')) {
    log (" [FAIL]");
    log (" ** Session initialization failed: Model path was not provided");
    return false;
  }
  session = std::make_unique< Ort::Session > (
    getEnv(),
    modelPath,
    sessionOptions
  );
  log (" [OK]");

  // Getting input/output nodes info
  log ("--> Getting model information...");
  num_input_nodes = getSession().GetInputCount();
  num_output_nodes = getSession().GetOutputCount();
  char buffer[100];
  sprintf(buffer, " * Number of inputs: %zu", num_input_nodes);
  log (buffer);
  for (short i = 0; i < num_input_nodes; i++) {
    sprintf(buffer, "  - Input #%d:", i);
    log (buffer);
    char *input_name = getSession().GetInputName(i, allocator);
    sprintf(buffer, "   . Name: %s", input_name);
    log (buffer);
    input_node_names.push_back(input_name);

    Ort::TypeInfo type_info = getSession().GetInputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType type = tensor_info.GetElementType();
    sprintf(buffer, "   . Type: %d", type);
    log(buffer);

    input_node_dims = tensor_info.GetShape();
    sprintf(buffer, "   . Number of dimensions: %zu", input_node_dims.size());
    log (buffer);
    for (short j = 0; j < input_node_dims.size(); j++) {
      sprintf(buffer, "      dim #%d -> %jd", j, input_node_dims[j]);
      log (buffer);
    }
  }
  sprintf(buffer, " * Number of outputs: %zu", num_output_nodes);
  log (buffer);
  for (short i = 0; i < num_output_nodes; i++) {
    sprintf(buffer, "  - Output #%d:", i);
    log (buffer);
    char *output_name = getSession().GetOutputName(i, allocator);
    sprintf(buffer, "   . Name: %s", output_name);
    log (buffer);
    output_node_names.push_back(output_name);

    Ort::TypeInfo type_info = getSession().GetOutputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType type = tensor_info.GetElementType();
    sprintf(buffer, "   . Type: %d", type);
    log(buffer);

    output_node_dims = tensor_info.GetShape();
    sprintf(buffer, "   . Number of dimensions: %zu", output_node_dims.size());
    log (buffer);
    for (short j = 0; j < output_node_dims.size(); j++) {
      sprintf(buffer, "      dim #%d -> %jd", j, output_node_dims[j]);
      log (buffer);
    }
  }

  // Fix for batch size dimension:
  // https://github.com/microsoft/onnxruntime/issues/3258
  input_node_dims[0] = batch_size;
  output_node_dims[0] = batch_size;

  // Exiting
  log ("--> Initialization is complete!");

  return true;
}

/*
 *  Executing
 */
bool ORTInference::execute () {

  log ("--> Entering execution...");

  // Generate dummy input for testing
  log ("--> Generating dummy data for testing...", "");
  size_t input_tensor_size = 224 * 224 * 3;
  std::vector<float> input_tensor_values(input_tensor_size);
  for (unsigned int i = 0; i < input_tensor_size; i++)
    input_tensor_values[i] = (float)i / (input_tensor_size + 1);
  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size, input_node_dims.data(), 4);
  assert(input_tensor.IsTensor());
  log (" [OK]");

  // Scoring that dummy data
  log ("--> Scoring data...", "");
  auto output_tensors = getSession().Run(Ort::RunOptions{nullptr}, input_node_names.data(), &input_tensor, 1, output_node_names.data(), 1);
  assert(output_tensors.size() == 1 && output_tensors.front().IsTensor());
  float* floatarr = output_tensors.front().GetTensorMutableData<float>();
  log (" [OK]");

  // Printing scores
  // (won't use log function here since I intend to delete this later)
  for (int i = 0; i < 5; i++)
    printf(" * Score for class %d:  %f\n", i, floatarr[i]);

  log ("--> Execution is complete!");
}

/*
 *  Finalization
 */
bool ORTInference::finalize () {
  log ("--> Finalizing...", "");
  log (" [OK]");
}

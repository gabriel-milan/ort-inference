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
  input_tensor_size = 1;
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
      if (j > 0) input_tensor_size *= input_node_dims[j];
    }
  }
  sprintf(buffer, "   . Input tensor size: %zu", input_tensor_size);
  log(buffer);
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
  // Execution code goes here
  log ("--> Execution is complete!");
  
}

/*
 *  Finalization
 */
bool ORTInference::finalize () {
  log ("--> Finalizing...", "");
  // Finalization code goes here
  log (" [OK]");
  return true;
}

/*
 *  Making predictions
 */
float *ORTInference::predict (std::vector<float> sample_values) {

  log ("Converting input to tensor...", "");
  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
    memory_info,
    sample_values.data(),
    input_tensor_size,
    input_node_dims.data(),
    input_node_dims.size()
  );
  assert(input_tensor.IsTensor());
  log (" [OK]");

  log ("Predicting...", "");
  auto output_tensors = getSession().Run(Ort::RunOptions{nullptr}, input_node_names.data(), &input_tensor, 1, output_node_names.data(), 1);
  assert(output_tensors.size() == 1 && output_tensors.front().IsTensor());
  float* floatarr = output_tensors.front().GetTensorMutableData<float>();
  log (" [OK]");

  return floatarr;
}

/*
 *  Changing optimization parameters
 */
void ORTInference::setGraphOptimizationLevel (GraphOptimizationLevel level) {
  sessionOptions.SetGraphOptimizationLevel(level);
}
/*
 *  Author: Gabriel Gazola Milan
 *  Date:   July 15, 2020
 *  Licensed under the MIT license
 */

#ifndef ORT_INFERENCE_H
#define ORT_INFERENCE_H

#include <assert.h>
#include <vector>
#include <onnxruntime_cxx_api.h>

#define DEFAULT_OPTIMIZATION_LEVEL      GraphOptimizationLevel::ORT_DISABLE_ALL
#define DEFAULT_LOGGING_LEVEL           ORT_LOGGING_LEVEL_WARNING
#define DEFAULT_ENV_NAME                "ort_inference"

class ORTInference {

  public:

    ORTInference ();
    ORTInference (const char *modelPath);

    bool initialize ();
    bool execute ();
    bool finalize ();

    float *predict (std::vector<float> sample_values);

    // Default setters/getters
    virtual Ort::Env &getEnv() const { return *env; };
    virtual Ort::Session &getSession() const { return *session; };

    char *getEnvName () { return envName; };
    void setEnvName (const char *name) { envName = (char *)name; };

    int64_t getBatchSize () { return batch_size; };
    void setBatchSize (int64_t size) { batch_size = size; };

    char *getModelPath () { return modelPath; };
    void setModelPath (const char *path) { modelPath = (char *)path; };

    OrtLoggingLevel getLoggingLevel () { return loggingLevel; };
    void setLoggingLevel (OrtLoggingLevel level) { loggingLevel = level; };

    void setGraphOptimizationLevel (GraphOptimizationLevel level);


  private:

    // ORT specific
    std::unique_ptr< Ort::Env > env;
    std::unique_ptr< Ort::Session > session;
    Ort::SessionOptions sessionOptions;
    Ort::AllocatorWithDefaultOptions allocator;

    // Model characteristics
    char *modelPath=(char *)"";
    int64_t batch_size=1;

    // Initialization
    char *envName=(char *)DEFAULT_ENV_NAME;
    OrtLoggingLevel loggingLevel=DEFAULT_LOGGING_LEVEL;

    // Logging
    bool console = true;
    void log (const char *message, const char *end);

    // Inner work
    size_t num_input_nodes;
    size_t num_output_nodes;
    size_t input_tensor_size;
    std::vector<int64_t> input_node_dims;
    std::vector<int64_t> output_node_dims;
    std::vector<const char*> input_node_names;
    std::vector<const char*> output_node_names;

};

#endif

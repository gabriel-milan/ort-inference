#ifndef ORT_INFERENCE_H
#define ORT_INFERENCE_H

class ORTInference {

  public:

    ORTInference ();
    ORTInference (char *modelPath);

    void setGraphOptimizationLevel (GraphOptimizationLevel level);


  private:

    Ort::Env env();
    Ort::Session session;
    Ort::SessionOptions sessionOptions;
    char *modelPath;

#endif

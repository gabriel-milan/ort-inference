# ORT Inference

Edit `ringer.cxx` or `vgg16.cxx` to attend your needs. They're both examples.

* `ringer.cxx` runs the NeuralRinger v8 in ONNX format;
* `vgg16.cxx` runs the VGG16 in ONNX format;

## Test it

```
make ringer
make run_ringer
```

Sample output for Ringer:

```
--> Initializing environment... [OK]
--> Initializing session... [OK]
--> Getting model information...
 * Number of inputs: 1
  - Input #0:
   . Name: input_1
   . Type: 1
   . Number of dimensions: 2
      dim #0 -> -1
      dim #1 -> 100
   . Input tensor size: 100
 * Number of outputs: 1
  - Output #0:
   . Name: activation_1
   . Type: 1
   . Number of dimensions: 2
      dim #0 -> -1
      dim #1 -> 1
--> Initialization is complete!
 - Iteration #0
Converting input to tensor... [OK]
Predicting... [OK]
   . Score: 0.585945
 - Iteration #1
Converting input to tensor... [OK]
Predicting... [OK]
   . Score: 0.585945
 - Iteration #2
Converting input to tensor... [OK]
Predicting... [OK]
   . Score: 0.585945
 - Iteration #3
Converting input to tensor... [OK]
Predicting... [OK]
   . Score: 0.585945
 - Iteration #4
Converting input to tensor... [OK]
Predicting... [OK]
   . Score: 0.585945
 - Iteration #5
Converting input to tensor... [OK]
Predicting... [OK]
   . Score: 0.585945
 - Iteration #6
Converting input to tensor... [OK]
Predicting... [OK]
   . Score: 0.585945
 - Iteration #7
Converting input to tensor... [OK]
Predicting... [OK]
   . Score: 0.585945
 - Iteration #8
Converting input to tensor... [OK]
Predicting... [OK]
   . Score: 0.585945
 - Iteration #9
Converting input to tensor... [OK]
Predicting... [OK]
   . Score: 0.585945
--> Finalizing... [OK]
```
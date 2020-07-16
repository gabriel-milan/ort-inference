# ORT Inference

Edit `main.cxx` to attend your needs. Still in development.

```
make
./main
```

Sample output for VGG16:

```
--> Initializing environment... [OK]
--> Initializing session... [OK]
--> Getting model information...
 * Number of inputs: 1
  - Input #0:
   . Name: input_1
   . Type: 1
   . Number of dimensions: 4
      dim #0 -> -1
      dim #1 -> 224
      dim #2 -> 224
      dim #3 -> 3
 * Number of outputs: 1
  - Output #0:
   . Name: predictions
   . Type: 1
   . Number of dimensions: 2
      dim #0 -> -1
      dim #1 -> 1000
--> Initialization is complete!
--> Entering execution...
--> Generating dummy data for testing... [OK]
--> Scoring data... [OK]
 * Score for class 0:  0.000181
 * Score for class 1:  0.002160
 * Score for class 2:  0.000448
 * Score for class 3:  0.000764
 * Score for class 4:  0.001572
--> Execution is complete!
--> Finalizing... [OK]
```
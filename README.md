# PyTorch_Quantization
all methods of pytorch quantization based on resnet50 with cifar-10

# Method
User should run 
```
python3 quantization.py --tq [BOOL] --sq [BOOL] --qat [BOOL]
```
Each argument parser means

tq : tutorial qauntization, which imports quantized model where pytorch official page offers

sq : static quantization, manually defines resnet 50 models and quantize

qat : quantization aware training, train with illusive transformer (fp32 -> int8) while training

# Future work
- [ ] quantization aware training
  
  need more training epochs for training code
- [ ] speed issue in CPU
  
  currently quantized model is more slower than expected, need to test in mobile devices
  
- [ ] various backbone model

  currently supporting resnet only
  
- [ ] customized dataset loading


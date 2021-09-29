import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

import copy

from model.resnet import resnet18, resnet50
from model.quantized_resnet import QuantizedResNet
from utils import set_random_seeds, model_equivalence, measure_inference_latency, print_model_size, run_benchmark
from data import prepare_dataloader

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-tq', '--tutorial_quantization', type=bool, default=False)
    parser.add_argument('-sq', '--static_quantization', type=bool, default=False)
    parser.add_argument('-qat', '--quantization_aware_training', type=bool, default=False)
    args = parser.parse_args()

    return args

def evaluate_model(model, test_loader, device, criterion=None):
    model.eval()
    model.to(device)

    running_loss = 0
    running_corrects = 0

    for inputs, labels in test_loader:

        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        if criterion is not None:
            loss = criterion(outputs, labels).item()
        else:
            loss = 0

        # statistics
        running_loss += loss * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    eval_loss = running_loss / len(test_loader.dataset)
    eval_accuracy = running_corrects / len(test_loader.dataset)

    return eval_loss, eval_accuracy

def train_model(model, train_loader, test_loader, device, learning_rate=1e-1, num_epochs=10):
    # The training configurations were not carefully selected.
    criterion = nn.CrossEntropyLoss()

    model.to(device)

    # It seems that SGD optimizer is better than Adam optimizer for ResNet18 training on CIFAR10.
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-5)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=[100, 150],
                                                     gamma=0.1,
                                                     last_epoch=-1)

    for epoch in range(num_epochs):

        # Training
        model.train()

        running_loss = 0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        train_loss = running_loss / len(train_loader.dataset)
        train_accuracy = running_corrects / len(train_loader.dataset)

        # Evaluation
        model.eval()
        eval_loss, eval_accuracy = evaluate_model(model=model, test_loader=test_loader, device=device,
                                                  criterion=criterion)

        # Set learning rate scheduler
        scheduler.step()

        print("Epoch: {:02d} Train Loss: {:.3f} Train Acc: {:.3f} Eval Loss: {:.3f} Eval Acc: {:.3f}".format(epoch,
                                                                                                             train_loss,
                                                                                                             train_accuracy,
                                                                                                             eval_loss,
                                                                                                             eval_accuracy))

    return model

def calibrate_model(model, loader, device=torch.device("cpu:0")):
    model.to(device)
    model.eval()

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        _ = model(inputs)

def save_model(model, model_dir, model_filename):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_filepath = os.path.join(model_dir, model_filename)
    torch.save(model.state_dict(), model_filepath)

def load_model(model, model_filepath, device):
    model.load_state_dict(torch.load(model_filepath, map_location=device))

    return model

def save_torchscript_model(model, model_dir, model_filename):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_filepath = os.path.join(model_dir, model_filename)
    torch.jit.save(torch.jit.script(model), model_filepath)


def load_torchscript_model(model_filepath, device):
    model = torch.jit.load(model_filepath, map_location=device)

    return model

def create_model(num_classes=10):
    # The number of channels in ResNet18 is divisible by 8.
    # This is required for fast GEMM integer matrix multiplication.
    # model = torchvision.models.resnet18(pretrained=False)
    # model = resnet18(num_classes=num_classes, pretrained=False)
    model = resnet50(num_classes=num_classes, pretrained=False)

    # We would use the pretrained ResNet18 as a feature extractor.
    # for param in model.parameters():
    #     param.requires_grad = False

    # Modify the last FC layer
    # num_features = model.fc.in_features
    # model.fc = nn.Linear(num_features, 10)

    return model

def gpu():
    random_seed = 0
    num_classes = 10
    cpu_device = torch.device("cpu:0")
    if torch.cuda.is_available():
        cuda_device = torch.device("cuda:0")
    else:
        assert AssertionError

    model_dir = "saved_models"
    model_filename = "resnet50_cifar10.pt"
    tutorial_quantized_model_filename = "resnet50_tutorial_quantized_imagenet.pt"
    static_jit_model_filename = "resnet50_static_jit_cifar10.pt"
    static_quantized_model_filename = "resnet50_static_quantized_cifar10.pt"
    qat_quantized_model_filename = "resnet50_qat_quantized_cifar10.pt"
    model_filepath = os.path.join(model_dir, model_filename)
    tutorial_quantized_model_filepath = os.path.join(model_dir, tutorial_quantized_model_filename)
    static_jit_model_filepath = os.path.join(model_dir, static_jit_model_filename)
    static_quantized_model_filepath = os.path.join(model_dir, static_quantized_model_filename)
    qat_quantized_model_filepath = os.path.join(model_dir, qat_quantized_model_filename)

    set_random_seeds(random_seed=random_seed)

    # argument parse and create log
    args = parse_args()

    # Create an untrained model.
    model = create_model(num_classes=num_classes)

    train_loader, test_loader = prepare_dataloader(num_workers=8, train_batch_size=128, eval_batch_size=256)

    if not os.path.isfile(model_filepath):
        # Train model.
        model = train_model(model=model, train_loader=train_loader, test_loader=test_loader, device=cuda_device)
        # Save model.
        save_model(model=model, model_dir=model_dir, model_filename=model_filename)

    # Load a pretrained model.
    model = load_model(model=model, model_filepath=model_filepath, device=cuda_device)
    # Move the model to CPU since static quantization does not support CUDA currently.
    model.to(cpu_device)
    # Make a copy of the model for layer fusion
    fused_model = copy.deepcopy(model)

    model.eval()
    # The model has to be switched to evaluation mode before any layer fusion.
    # Otherwise the quantization will not work correctly.
    fused_model.eval()

    _, fp32_eval_accuracy = evaluate_model(model=model, test_loader=test_loader, device=cpu_device, criterion=None)

    if args.tutorial_quantization == True:
        tutorial_model = torchvision.models.resnet50(pretrained=True)
        tutorial_quantized_model = torchvision.models.quantization.resnet50(pretrained=True, quantize=True)

        print_model_size(tutorial_model)
        print_model_size(tutorial_quantized_model)

        save_model(model=tutorial_quantized_model, model_dir=model_dir, model_filename=tutorial_quantized_model_filename)

        # Load quantized model.
        # tutorial_quantized_jit_model = load_torchscript_model(model_filepath=tutorial_quantized_model_filepath,
        #                                                     device=cpu_device)

        tutorial_fp32_cpu_inference_latency = measure_inference_latency(model=tutorial_model, device=cpu_device,
                                                               input_size=(1, 3, 256, 256),
                                                               num_samples=100)
        tutorial_fp32_gpu_inference_latency = measure_inference_latency(model=tutorial_model, device=cuda_device,
                                                               input_size=(1, 3, 256, 256), num_samples=100)

        tutorial_int8_cpu_inference_latency = measure_inference_latency(model=tutorial_quantized_model, device=cpu_device,
                                                               input_size=(1, 3, 256, 256), num_samples=100)
        # tutorial_int8_jit_cpu_inference_latency = measure_inference_latency(model=tutorial_quantized_jit_model, device=cpu_device,
        #                                                            input_size=(1, 3, 256, 256), num_samples=100)

        print("*** Result of Tutorial Quantization ***")
        print("FP32 CPU Inference Latency: {:.2f} ms / sample".format(tutorial_fp32_cpu_inference_latency * 1000))
        print("FP32 CUDA Inference Latency: {:.2f} ms / sample".format(tutorial_fp32_gpu_inference_latency * 1000))
        print("INT8 CPU Inference Latency: {:.2f} ms / sample".format(tutorial_int8_cpu_inference_latency * 1000))
        # print("INT8 JIT CPU Inference Latency: {:.2f} ms / sample".format(tutorial_int8_jit_cpu_inference_latency * 1000))

    if args.static_quantization == True:
        # Fuse the model in place rather manually.
        static_model = torch.quantization.fuse_modules(fused_model, [["conv1", "bn1", "relu"]], inplace=True)
        for module_name, module in static_model.named_children():
            if "layer" in module_name:
                for basic_block_name, basic_block in module.named_children():
                    torch.quantization.fuse_modules(basic_block, [["conv1", "bn1", "relu1"], ["conv2", "bn2", "relu2"], ["conv3", "bn3"]],
                                                    inplace=True)
                    for sub_block_name, sub_block in basic_block.named_children():
                        if sub_block_name == "downsample":
                            torch.quantization.fuse_modules(sub_block, [["0", "1"]], inplace=True)

        # Print FP32 model.
        print(model)
        # Print fused model.
        print(static_model)

        # Model and fused model should be equivalent.
        assert model_equivalence(model_1=model, model_2=static_model, device=cpu_device, rtol=1e-03, atol=1e-06,
                                 num_tests=100,
                                 input_size=(1, 3, 32, 32)), "Fused model is not equivalent to the original model!"

        # Prepare the model for static quantization. This inserts observers in
        # the model that will observe activation tensors during calibration.
        static_quantized_model = QuantizedResNet(model_fp32=static_model)
        # Using un-fused model will fail.
        # Because there is no quantized layer implementation for a single batch normalization layer.
        # static_quantized_model = QuantizedResNet18(model_fp32=model)
        # Select quantization schemes from
        # https://pytorch.org/docs/stable/quantization-support.html
        quantization_config = torch.quantization.get_default_qconfig("fbgemm")
        # Custom quantization configurations
        # quantization_config = torch.quantization.default_qconfig
        # quantization_config = torch.quantization.QConfig(activation=torch.quantization.MinMaxObserver.with_args(dtype=torch.quint8), weight=torch.quantization.MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric))

        static_quantized_model.qconfig = quantization_config

        # Print quantization configurations
        print(static_quantized_model.qconfig)

        torch.quantization.prepare(static_quantized_model, inplace=True)

        # Use training data for calibration.
        calibrate_model(model=static_quantized_model, loader=train_loader, device=cpu_device)

        static_quantized_model = torch.quantization.convert(static_quantized_model, inplace=True)

        # Using high-level static quantization wrapper
        # The above steps, including torch.quantization.prepare, calibrate_model, and torch.quantization.convert, are also equivalent to
        # static_quantized_model = torch.quantization.quantize(model=static_quantized_model, run_fn=calibrate_model, run_args=[train_loader], mapping=None, inplace=False)

        static_quantized_model.eval()

        # Print quantized model.
        print(static_quantized_model)

        # Save quantized model.
        save_torchscript_model(model=model, model_dir=model_dir,
                               model_filename=static_jit_model_filename)
        save_torchscript_model(model=static_quantized_model, model_dir=model_dir,
                               model_filename=static_quantized_model_filename)

        # Load quantized model.
        static_quantized_jit_model = load_torchscript_model(model_filepath=static_quantized_model_filepath,
                                                            device=cpu_device)

        _, static_int8_eval_accuracy = evaluate_model(model=static_quantized_jit_model, test_loader=test_loader,
                                               device=cpu_device,
                                               criterion=None)

        static_int8_cpu_inference_latency = measure_inference_latency(model=static_quantized_model, device=cpu_device,
                                                               input_size=(1, 3, 32, 32), num_samples=100)
        static_int8_jit_cpu_inference_latency = measure_inference_latency(model=static_quantized_jit_model, device=cpu_device,
                                                                   input_size=(1, 3, 32, 32), num_samples=100)

    if args.quantization_aware_training == True:
        # The model has to be switched to training mode before any layer fusion.
        # Otherwise the quantization aware training will not work correctly.
        fused_model.train()

        # Fuse the model in place rather manually.
        qat_model = torch.quantization.fuse_modules(fused_model, [["conv1", "bn1", "relu"]], inplace=True)
        for module_name, module in qat_model.named_children():
            if "layer" in module_name:
                for basic_block_name, basic_block in module.named_children():
                    torch.quantization.fuse_modules(basic_block, [["conv1", "bn1", "relu1"], ["conv2", "bn2", "relu2"], ["conv3", "bn3"]],
                                                    inplace=True)
                    for sub_block_name, sub_block in basic_block.named_children():
                        if sub_block_name == "downsample":
                            torch.quantization.fuse_modules(sub_block, [["0", "1"]], inplace=True)

        # Print FP32 model.
        print(model)
        # Print fused model.
        print(qat_model)

        # equivalence will not pass if model is not eval mode
        qat_model.eval()

        # Model and fused model should be equivalent.
        assert model_equivalence(model_1=model, model_2=qat_model, device=cpu_device, rtol=1e-03, atol=1e-06,
                                 num_tests=100,
                                 input_size=(1, 3, 32, 32)), "Fused model is not equivalent to the original model!"

        # Prepare the model for static quantization. This inserts observers in
        # the model that will observe activation tensors during calibration.
        qat_quantized_model = QuantizedResNet(model_fp32=qat_model)
        # Using un-fused model will fail.
        # Because there is no quantized layer implementation for a single batch normalization layer.
        # qat_quantized_model = QuantizedResNet18(model_fp32=model)
        # Select quantization schemes from
        # https://pytorch.org/docs/stable/quantization-support.html
        quantization_config = torch.quantization.get_default_qconfig("fbgemm")
        # Custom quantization configurations
        # quantization_config = torch.quantization.default_qconfig
        # quantization_config = torch.quantization.QConfig(activation=torch.quantization.MinMaxObserver.with_args(dtype=torch.quint8), weight=torch.quantization.MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric))

        qat_quantized_model.qconfig = quantization_config

        # Print quantization configurations
        print(qat_quantized_model.qconfig)

        # https://pytorch.org/docs/stable/_modules/torch/quantization/quantize.html#prepare_qat
        torch.quantization.prepare_qat(qat_quantized_model, inplace=True)
        # # Use training data for calibration.
        print("Training QAT Model...")
        qat_quantized_model.train()
        train_model(model=qat_quantized_model,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    device=cuda_device,
                    learning_rate=1e-3,
                    num_epochs=10)

        qat_quantized_model.to(cpu_device)

        qat_quantized_model = torch.quantization.convert(qat_quantized_model, inplace=True)

        qat_quantized_model.eval()

        # Print quantized model.
        print(qat_quantized_model)

        # Save quantized model.
        save_torchscript_model(model=qat_quantized_model,
                               model_dir=model_dir,
                               model_filename=qat_quantized_model_filename)

        # Load quantized model.
        qat_quantized_jit_model = load_torchscript_model(
            model_filepath=qat_quantized_model_filepath, device=cpu_device)

        _, qat_int8_eval_accuracy = evaluate_model(model=qat_quantized_jit_model, test_loader=test_loader,
                                               device=cpu_device,
                                               criterion=None)

        qat_int8_cpu_inference_latency = measure_inference_latency(model=qat_quantized_model, device=cpu_device,
                                                               input_size=(1, 3, 32, 32), num_samples=100)
        qat_int8_jit_cpu_inference_latency = measure_inference_latency(model=qat_quantized_jit_model, device=cpu_device,
                                                                   input_size=(1, 3, 32, 32), num_samples=100)

    print("FP32 evaluation accuracy: {:.3f}".format(fp32_eval_accuracy))
    if args.static_quantization == True:
        print("*** Result of Static Quantization ***")
        print("INT8 evaluation accuracy: {:.3f}".format(static_int8_eval_accuracy))
    if args.quantization_aware_training == True:
        print("*** Result of Quantization Aware Training ***")
        print("INT8 evaluation accuracy: {:.3f}".format(qat_int8_eval_accuracy))
    print("")

    fp32_cpu_inference_latency = measure_inference_latency(model=model, device=cpu_device, input_size=(1, 3, 32, 32),
                                                           num_samples=100)
    fp32_gpu_inference_latency = measure_inference_latency(model=model, device=cuda_device,
                                                               input_size=(1, 3, 32, 32), num_samples=100)
    print("FP32 CPU Inference Latency: {:.2f} ms / sample".format(fp32_cpu_inference_latency * 1000))
    print("FP32 CUDA Inference Latency: {:.2f} ms / sample".format(fp32_gpu_inference_latency * 1000))

    if args.static_quantization == True:
        print("*** Result of Static Quantization ***")
        print("INT8 CPU Inference Latency: {:.2f} ms / sample".format(static_int8_cpu_inference_latency * 1000))
        print("INT8 JIT CPU Inference Latency: {:.2f} ms / sample".format(static_int8_jit_cpu_inference_latency * 1000))

        run_benchmark(static_jit_model_filepath, test_loader)
        run_benchmark(static_quantized_model_filepath, test_loader)
    if args.quantization_aware_training == True:
        print("*** Result of Quantization Aware Training ***")
        print("INT8 CPU Inference Latency: {:.2f} ms / sample".format(qat_int8_cpu_inference_latency * 1000))
        print("INT8 JIT CPU Inference Latency: {:.2f} ms / sample".format(qat_int8_jit_cpu_inference_latency * 1000))


def cpu():
    model_dir = "saved_models"
    model_filename = "resnet18_cifar10.pt"
    static_quantized_model_filename = "resnet18_quantized_cifar10.pt"
    model_filepath = os.path.join(model_dir, model_filename)
    static_quantized_model_filepath = os.path.join(model_dir, static_quantized_model_filename)

    cpu_device = torch.device("cpu:0")

    _, test_loader = prepare_dataloader(num_workers=8, train_batch_size=128, eval_batch_size=256)

    # Load quantized model.
    static_quantized_jit_model = load_torchscript_model(model_filepath=static_quantized_model_filepath,
                                                        device=cpu_device)

    _, int8_eval_accuracy = evaluate_model(model=static_quantized_jit_model, test_loader=test_loader, device=cpu_device,
                                           criterion=None)

    # Skip this assertion since the values might deviate a lot.
    # assert model_equivalence(model_1=model, model_2=quantized_jit_model, device=cpu_device, rtol=1e-01, atol=1e-02, num_tests=100, input_size=(1,3,32,32)), "Quantized model deviates from the original model too much!"

    print("INT8 evaluation accuracy: {:.3f}".format(int8_eval_accuracy))

    int8_jit_cpu_inference_latency = measure_inference_latency(model=static_quantized_jit_model, device=cpu_device,
                                                               input_size=(1, 3, 32, 32), num_samples=100)

    print("INT8 JIT CPU Inference Latency: {:.2f} ms / sample".format(int8_jit_cpu_inference_latency * 1000))

def main():
    if torch.cuda.is_available():
        gpu()
    else:
        cpu()

if __name__ == "__main__":
    main()
# WinML EP Benchmarking Sample App

A Windows desktop app built with **WinUI 3** and **Windows ML** that classifies images using the **ResNet-50** ONNX model. Pick any image and get the top-5 predictions with confidence scores.

## Prerequisites

- [.NET 8 SDK](https://dotnet.microsoft.com/download/dotnet/8.0)
- Windows 10 version 1809+ or Windows 11
- Visual Studio 2022 (recommended) or VS Code with C# Dev Kit

## Setup

### 1. Download the ONNX Model

Download **resnet50-v2-7.onnx** from the ONNX Model Zoo and place it in the `Assets/` folder:

```
Assets/
  resnet50-v2-7.onnx      <-- download this
  imagenet_labels.txt      <-- already included
```

Download link: https://github.com/onnx/models/blob/main/validated/vision/classification/resnet/model/resnet50-v2-7.onnx

### 2. Build & Run

```bash
dotnet restore
dotnet build
dotnet run
```

Or open in Visual Studio / VS Code and press F5.

## Usage

1. Click **"Pick an Image"** to select a `.jpg`, `.png`, or `.bmp` file
2. The app runs the image through ResNet-50 via Windows ML
3. Top-5 predicted labels and confidence scores are displayed

## Project Structure

| File | Description |
|------|-------------|
| `App.xaml` / `App.xaml.cs` | Application entry point |
| `MainWindow.xaml` / `MainWindow.xaml.cs` | UI with image picker and results display |
| `ImageClassifier.cs` | WinML inference logic using `LearningModelSession` |
| `Assets/resnet50-v2-7.onnx` | ResNet-50 ONNX model (user-provided) |
| `Assets/imagenet_labels.txt` | 1000 ImageNet class labels |

## Architecture

- **Microsoft.AI.MachineLearning** (WinML) for ONNX model loading and inference
- **Windows App SDK** (WinUI 3) for the modern desktop UI
- Image is decoded to `SoftwareBitmap` → `VideoFrame` for model input
- Output tensor is sorted to extract top-5 predictions

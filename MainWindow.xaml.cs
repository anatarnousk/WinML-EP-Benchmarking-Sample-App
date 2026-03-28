using Microsoft.UI.Xaml;
using Windows.Storage.Pickers;
using Microsoft.UI.Xaml.Media.Imaging;
using System.Collections.ObjectModel;

namespace WinMLResNet;

public sealed partial class MainWindow : Window
{
    private readonly ImageClassifier _classifier;
    private readonly ObservableCollection<PredictionResult> _results = new();
    private readonly ObservableCollection<ClassificationMetrics> _metrics = new();
    private string? _currentImagePath;
    private List<EpDeviceInfo> _devices = new();

    public MainWindow()
    {
        this.InitializeComponent();
        ResultsListView.ItemsSource = _results;
        MetricsListView.ItemsSource = _metrics;

        var modelPath = Path.Combine(AppContext.BaseDirectory, "Assets", "resnet50-v2-7.onnx");
        _classifier = new ImageClassifier(modelPath);

        PopulateModelInfo();
        PopulateRuntimeInfo();

        StatusText.Text = "Initializing Windows ML...";
        _ = InitializeClassifierAsync();
    }

    private void PopulateModelInfo()
    {
        try
        {
            var info = _classifier.GetModelInfo();
            ModelFileName.Text = info.FileName;
            ModelFileSize.Text = info.FileSize;
            ModelProducer.Text = info.ProducerName;
            ModelVersion.Text = info.ModelVersion.ToString();
            ModelQuantized.Text = info.IsQuantized ? $"Yes ({info.QuantizationDetail})" : info.QuantizationDetail;
            ModelDomain.Text = string.IsNullOrEmpty(info.Domain) ? "—" : info.Domain;
            ModelDescription.Text = string.IsNullOrEmpty(info.Description) ? "—" : info.Description;
            TensorListView.ItemsSource = info.Tensors;
        }
        catch (Exception ex)
        {
            ModelFileName.Text = $"Error: {ex.Message}";
        }
    }

    private void PopulateRuntimeInfo()
    {
        var rt = ImageClassifier.GetRuntimeInfo();
        VersionWAS.Text = rt.WindowsAppSdkVersion;
        VersionORT.Text = rt.OnnxRuntimeVersion;
        VersionWinML.Text = rt.WindowsMlVersion;
        VersionDotNet.Text = rt.DotNetVersion;
        VersionOS.Text = rt.OsVersion;
        VersionArch.Text = rt.Architecture;
        DeployMlApi.Text = rt.WindowsMlApi;
        DeployWasMode.Text = rt.WasdkDeployment;
        DeployEpMode.Text = rt.EpAcquisition;
    }

    private async Task InitializeClassifierAsync()
    {
        try
        {
            await _classifier.InitializeAsync();
            _devices = _classifier.GetAvailableDevices();
            EpSelector.ItemsSource = _devices;
            if (_devices.Count > 0)
                EpSelector.SelectedIndex = 0;
            StatusText.Text = $"Ready — {_devices.Count} EP device(s) available.";
        }
        catch (Exception ex)
        {
            StatusText.Text = $"Init error: {ex.Message}";
        }
    }

    private async void OnPickImage(object sender, RoutedEventArgs e)
    {
        var picker = new FileOpenPicker();
        picker.FileTypeFilter.Add(".jpg");
        picker.FileTypeFilter.Add(".jpeg");
        picker.FileTypeFilter.Add(".png");
        picker.FileTypeFilter.Add(".bmp");

        var hwnd = WinRT.Interop.WindowNative.GetWindowHandle(this);
        WinRT.Interop.InitializeWithWindow.Initialize(picker, hwnd);

        var file = await picker.PickSingleFileAsync();
        if (file is null) return;

        _currentImagePath = file.Path;

        // Show preview
        var bitmapImage = new BitmapImage();
        using (var stream = await file.OpenReadAsync())
        {
            await bitmapImage.SetSourceAsync(stream);
        }
        PreviewImage.Source = bitmapImage;

        ClassifyButton.IsEnabled = true;
        RunAllButton.IsEnabled = true;
        StatusText.Text = $"Image loaded: {file.Name}";
    }

    private async void OnClassify(object sender, RoutedEventArgs e)
    {
        if (_currentImagePath is null || EpSelector.SelectedItem is not EpDeviceInfo device) return;

        SetButtonsEnabled(false);
        StatusText.Text = $"Classifying with {device.DisplayName}...";

        try
        {
            // Run uncompiled
            var result = await _classifier.ClassifyAsync(_currentImagePath, device, compiled: false);

            _results.Clear();
            foreach (var p in result.Predictions)
                _results.Add(p);

            // Update metrics — replace existing entries for this EP
            var existingUncompiled = _metrics.FirstOrDefault(m => m.EpName == device.DisplayName && m.Mode == "Uncompiled");
            if (existingUncompiled != null) _metrics.Remove(existingUncompiled);
            _metrics.Add(result.Metrics);

            // Run compiled
            StatusText.Text = $"Compiling & classifying with {device.DisplayName}...";
            try
            {
                var compiledResult = await _classifier.ClassifyAsync(_currentImagePath, device, compiled: true);
                var existingCompiled = _metrics.FirstOrDefault(m => m.EpName == device.DisplayName && m.Mode == "Compiled");
                if (existingCompiled != null) _metrics.Remove(existingCompiled);
                _metrics.Add(compiledResult.Metrics);
            }
            catch (Exception compEx)
            {
                _metrics.Add(new ClassificationMetrics
                {
                    EpName = device.DisplayName,
                    Mode = "Compiled",
                    CompileTime = "Error",
                    InferenceTime = "—",
                    TotalTime = compEx.Message
                });
            }

            StatusText.Text = $"Done — {device.DisplayName}: {result.Metrics.InferenceTime} inference";
        }
        catch (Exception ex)
        {
            StatusText.Text = $"Error: {ex.Message}";
        }
        finally
        {
            SetButtonsEnabled(true);
        }
    }

    private async void OnRunAll(object sender, RoutedEventArgs e)
    {
        if (_currentImagePath is null) return;

        SetButtonsEnabled(false);
        _metrics.Clear();
        int total = _devices.Count * 2; // uncompiled + compiled per EP
        int step = 0;

        for (int i = 0; i < _devices.Count; i++)
        {
            var device = _devices[i];

            // Uncompiled
            step++;
            StatusText.Text = $"[{step}/{total}] {device.DisplayName} (uncompiled)...";
            try
            {
                var result = await _classifier.ClassifyAsync(_currentImagePath, device, compiled: false);
                _metrics.Add(result.Metrics);

                if (i == _devices.Count - 1)
                {
                    _results.Clear();
                    foreach (var p in result.Predictions)
                        _results.Add(p);
                }
            }
            catch (Exception ex)
            {
                _metrics.Add(new ClassificationMetrics
                {
                    EpName = device.DisplayName,
                    Mode = "Uncompiled",
                    InferenceTime = "Error",
                    TotalTime = ex.Message
                });
            }

            // Compiled
            step++;
            StatusText.Text = $"[{step}/{total}] {device.DisplayName} (compiled)...";
            try
            {
                var compiledResult = await _classifier.ClassifyAsync(_currentImagePath, device, compiled: true);
                _metrics.Add(compiledResult.Metrics);
            }
            catch (Exception ex)
            {
                _metrics.Add(new ClassificationMetrics
                {
                    EpName = device.DisplayName,
                    Mode = "Compiled",
                    CompileTime = "Error",
                    InferenceTime = "—",
                    TotalTime = ex.Message
                });
            }
        }

        StatusText.Text = $"Comparison complete — {_devices.Count} EP(s) x 2 modes benchmarked.";
        SetButtonsEnabled(true);
    }

    private void SetButtonsEnabled(bool enabled)
    {
        PickImageButton.IsEnabled = enabled;
        ClassifyButton.IsEnabled = enabled && _currentImagePath != null;
        RunAllButton.IsEnabled = enabled && _currentImagePath != null;
    }
}

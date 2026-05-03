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

        // WASDK 2.0 Pivot doesn't constrain content height properly — fix ScrollViewer on resize
        this.SizeChanged += (s, e) =>
        {
            // WASDK 2.0 Pivot doesn't constrain content height properly — fix ScrollViewer on resize
            var availableHeight = e.Size.Height - 160;
            if (availableHeight > 100)
                ClassificationScrollViewer.MaxHeight = availableHeight;

            // Cap image preview at 32% of window height
            ImagePreviewGrid.MaxHeight = e.Size.Height * 0.32;
        };

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
                    TotalTime = ExtractErrorCode(compEx)
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
                ApplyInferenceHeatMap();

                // Update predictions after every EP so they appear immediately
                _results.Clear();
                foreach (var p in result.Predictions)
                    _results.Add(p);
            }
            catch (Exception ex)
            {
                _metrics.Add(new ClassificationMetrics
                {
                    EpName = device.DisplayName,
                    Mode = "Uncompiled",
                    InferenceTime = "Error",
                    TotalTime = ExtractErrorCode(ex)
                });
            }

            // Compiled
            step++;
            StatusText.Text = $"[{step}/{total}] {device.DisplayName} (compiled)...";
            try
            {
                var compiledResult = await _classifier.ClassifyAsync(_currentImagePath, device, compiled: true);
                _metrics.Add(compiledResult.Metrics);
                ApplyInferenceHeatMap();
            }
            catch (Exception ex)
            {
                _metrics.Add(new ClassificationMetrics
                {
                    EpName = device.DisplayName,
                    Mode = "Compiled",
                    CompileTime = "Error",
                    InferenceTime = "—",
                    TotalTime = ExtractErrorCode(ex)
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

    private static string ExtractErrorCode(Exception ex)
    {
        var msg = ex.Message;
        var match = System.Text.RegularExpressions.Regex.Match(msg, @"\[ErrorCode:(\w+)\]");
        if (match.Success)
            return $"\u274C Error ({match.Groups[1].Value})";
        var short_ = msg.Length > 30 ? msg[..30] + "…" : msg;
        return $"\u274C Error ({short_})";
    }

    private void ApplyInferenceHeatMap()
    {
        // Apply heat map emoji to compile, preprocess, and total columns
        ApplyHeatMapToColumn(
            m => m.RawCompileMs,
            (m, text) => m.CompileTime = text,
            m => m.RawCompileMs >= 0);
        ApplyHeatMapToColumn(
            m => m.RawPreprocessMs,
            (m, text) => m.PreprocessTime = text,
            m => m.RawPreprocessMs >= 0);
        ApplyHeatMapToColumn(
            m => m.RawInferenceMs,
            (m, text) => m.InferenceTime = text,
            m => m.RawInferenceMs >= 0);
        ApplyHeatMapToColumn(
            m => m.RawTotalMs,
            (m, text) => m.TotalTime = text,
            m => m.RawTotalMs >= 0);

        // Force ListView to re-render with NEW object instances (WinUI reuses containers for same refs)
        // Sorted fastest to slowest by total time; errors go to bottom
        int rank = 0;
        var src = _metrics
            .OrderBy(m => m.RawTotalMs < 0 ? 1 : 0)
            .ThenBy(m => m.RawTotalMs < 0 ? 0 : m.RawTotalMs)
            .Select(m => new ClassificationMetrics
            {
                Rank = ++rank,
                EpName = m.EpName,
                Mode = m.Mode,
                RawCompileMs = m.RawCompileMs,
                CompileTime = m.CompileTime,
                RawPreprocessMs = m.RawPreprocessMs,
                PreprocessTime = m.PreprocessTime,
                RawInferenceMs = m.RawInferenceMs,
                InferenceTime = m.InferenceTime,
                RawTotalMs = m.RawTotalMs,
                TotalTime = m.TotalTime,
                TopPrediction = m.TopPrediction,
                TopConfidence = m.TopConfidence
            })
            .ToList();
        _metrics.Clear();
        foreach (var item in src)
            _metrics.Add(item);
    }

    private static readonly string[] HeatEmojis = [
        "\U0001F7E2", // green - fastest
        "\U0001F535", // blue
        "\U0001F7E1", // yellow
        "\U0001F7E0", // orange
        "\U0001F534"  // red - slowest
    ];

    private void ApplyHeatMapToColumn(
        Func<ClassificationMetrics, double> getRaw,
        Action<ClassificationMetrics, string> setText,
        Func<ClassificationMetrics, bool> isValid)
    {
        var measured = _metrics.Where(isValid).ToList();
        if (measured.Count == 0) return;

        // Sort ascending (fastest first)
        var ranked = measured.OrderBy(m => getRaw(m)).ToList();
        int n = ranked.Count;

        // Build emoji assignment: 5 buckets [green, blue, yellow, orange, red]
        // Distribute n items across 5 emoji slots using round-robin-ish assignment
        for (int i = 0; i < n; i++)
        {
            int emojiIndex;
            if (n == 1)
            {
                emojiIndex = 0; // single item → green
            }
            else if (n <= 5)
            {
                // n items, n emojis: direct mapping
                // Spread evenly: index 0→green, last→red, interpolate between
                emojiIndex = (int)Math.Round(i * 4.0 / (n - 1));
            }
            else
            {
                // More than 5 items: first=green(0), last=red(4), distribute rest across 1-3
                if (i == 0)
                    emojiIndex = 0;
                else if (i == n - 1)
                    emojiIndex = 4;
                else
                    emojiIndex = 1 + (int)Math.Round((i - 1) * 2.0 / (n - 3));
            }
            if (emojiIndex > 4) emojiIndex = 4;

            setText(ranked[i], $"{HeatEmojis[emojiIndex]} {getRaw(ranked[i]):F1} ms");
        }
    }
}

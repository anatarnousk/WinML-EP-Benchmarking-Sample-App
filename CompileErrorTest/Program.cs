using Microsoft.ML.OnnxRuntime;
using Microsoft.Windows.AI.MachineLearning;

Console.WriteLine("=== CompileModel Error Reproduction ===\n");

// Initialize ORT environment FIRST
var envOptions = new EnvironmentCreationOptions
{
    logId = "CompileTest",
    logLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE
};
OrtEnv.CreateInstanceWithOptions(ref envOptions);
var ortEnv = OrtEnv.Instance();
Console.WriteLine("OrtEnv created.");

// Register EPs
var catalog = ExecutionProviderCatalog.GetDefault();
await catalog.EnsureAndRegisterCertifiedAsync();
Console.WriteLine("EPs registered.\n");

// Enumerate devices
var devices = ortEnv.GetEpDevices();
Console.WriteLine($"Found {devices.Count} EP devices:");
foreach (var d in devices)
    Console.WriteLine($"  {d.EpName} ({d.HardwareDevice.Type})");

// Model path - use the same model from the main app
var modelPath = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "Assets", "resnet50-v2-7.onnx"));
if (!File.Exists(modelPath))
{
    // try alternate path
    modelPath = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "Assets", "resnet50-v2-7.onnx"));
}
Console.WriteLine($"\nModel path: {modelPath}");
Console.WriteLine($"Model exists: {File.Exists(modelPath)}\n");

if (!File.Exists(modelPath))
{
    Console.WriteLine("ERROR: Model file not found. Exiting.");
    return;
}

// Test CompileModel on each AUTO EP
foreach (var device in devices)
{
    if (!device.EpName.Contains("AUTO"))
        continue;

    Console.WriteLine($"\n{'=',-60}");
    Console.WriteLine($"Testing CompileModel on: {device.EpName} ({device.HardwareDevice.Type})");
    Console.WriteLine($"{'=',-60}");

    var sessionOptions = new SessionOptions();
    sessionOptions.AppendExecutionProvider(ortEnv, new[] { device }, new Dictionary<string, string>());

    var epSafe = device.EpName.Replace(" ", "_").Replace(".", "_");
    var compiledPath = Path.Combine(Path.GetTempPath(), $"test-compiled-{epSafe}.onnx");

    // Clean up any previous compiled file
    if (File.Exists(compiledPath)) File.Delete(compiledPath);

    try
    {
        using var compileOptions = new OrtModelCompilationOptions(sessionOptions);
        compileOptions.SetInputModelPath(modelPath);
        compileOptions.SetOutputModelPath(compiledPath);
        compileOptions.CompileModel();
        Console.WriteLine("CompileModel SUCCEEDED (unexpected for AUTO EP)");
        if (File.Exists(compiledPath))
            Console.WriteLine($"  Compiled file size: {new FileInfo(compiledPath).Length} bytes");
    }
    catch (Exception ex)
    {
        Console.WriteLine($"\n--- Exception Type ---");
        Console.WriteLine(ex.GetType().FullName);
        Console.WriteLine($"\n--- HResult ---");
        Console.WriteLine($"0x{ex.HResult:X8}");
        Console.WriteLine($"\n--- Message ---");
        Console.WriteLine(ex.Message);
        Console.WriteLine($"\n--- Full ToString() ---");
        Console.WriteLine(ex.ToString());

        if (ex.InnerException != null)
        {
            Console.WriteLine($"\n--- Inner Exception Type ---");
            Console.WriteLine(ex.InnerException.GetType().FullName);
            Console.WriteLine($"\n--- Inner Exception Message ---");
            Console.WriteLine(ex.InnerException.Message);
            Console.WriteLine($"\n--- Inner Exception ToString() ---");
            Console.WriteLine(ex.InnerException.ToString());
        }
    }
}

Console.WriteLine("\n=== Done ===");

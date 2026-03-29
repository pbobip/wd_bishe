$ErrorActionPreference = "Stop"

$python = "C:\Users\pyd111\fy\Scripts\python.exe"
$projectRoot = "C:\Users\pyd111\Desktop\中期\wd_bishe"
$env:PYTHONUNBUFFERED = "1"
$configPaths = @(
    "C:\Users\pyd111\Desktop\中期\wd_bishe\experiments\mbu_netpp\configs\e2_micronet_edge_gpu.yaml",
    "C:\Users\pyd111\Desktop\中期\wd_bishe\experiments\mbu_netpp\configs\e2_micronet_edge_deep_gpu.yaml",
    "C:\Users\pyd111\Desktop\中期\wd_bishe\experiments\mbu_netpp\configs\e3_micronet_edge_deep_vf_gpu.yaml"
)

$logRoot = Join-Path $projectRoot "experiments\mbu_netpp\outputs\logs"
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null

foreach ($configPath in $configPaths) {
    $experimentName = [System.IO.Path]::GetFileNameWithoutExtension($configPath)
    $logPath = Join-Path $logRoot ($experimentName + ".log")
    Write-Host "==== 开始运行 $experimentName ===="
    & $python -u -m experiments.mbu_netpp.train --config $configPath --run-all-folds *>&1 | Tee-Object -FilePath $logPath
    if ($LASTEXITCODE -ne 0) {
        throw "实验失败: $experimentName"
    }
}

Write-Host "==== E2/E3 队列完成 ===="

$ErrorActionPreference = "Stop"

$python = "C:\Users\pyd111\fy\Scripts\python.exe"
$projectRoot = "C:\Users\pyd111\Desktop\中期\wd_bishe"
$env:PYTHONUNBUFFERED = "1"
$configPaths = @(
    "C:\Users\pyd111\Desktop\中期\wd_bishe\experiments\mbu_netpp\configs\e1a_unetpp_noaug_gpu.yaml",
    "C:\Users\pyd111\Desktop\中期\wd_bishe\experiments\mbu_netpp\configs\e1a_unetpp_aug_gpu.yaml",
    "C:\Users\pyd111\Desktop\中期\wd_bishe\experiments\mbu_netpp\configs\e1_unet_gpu.yaml",
    "C:\Users\pyd111\Desktop\中期\wd_bishe\experiments\mbu_netpp\configs\e1_unetpp_gpu.yaml",
    "C:\Users\pyd111\Desktop\中期\wd_bishe\experiments\mbu_netpp\configs\e1_micronet_unetpp_gpu.yaml"
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

Write-Host "==== 全部基线完成 ===="

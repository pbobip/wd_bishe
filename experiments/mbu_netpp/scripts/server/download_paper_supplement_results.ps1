param(
    [Parameter(Mandatory = $true)]
    [string]$RemoteHost,
    [string]$RemoteUser = "root",
    [int]$Port = 22,
    [string]$IdentityFile = "",
    [string]$RemoteRoot = "/root/wd_bishe_opt_suite",
    [string]$RemoteArchiveRelative = "results/server_delivery/paper_supplement_all.tar.gz",
    [string]$LocalOutputParent = "D:\中期\wd_bishe\results",
    [string]$ResultName = "paper_supplement_all"
)

$ErrorActionPreference = "Stop"

function Ensure-Dir {
    param([Parameter(Mandatory = $true)][string]$PathText)
    if (-not (Test-Path -LiteralPath $PathText)) {
        New-Item -ItemType Directory -Path $PathText | Out-Null
    }
}

function Invoke-Scp {
    param(
        [Parameter(Mandatory = $true)][string]$SourcePath,
        [Parameter(Mandatory = $true)][string]$DestinationPath
    )

    $args = @("-P", "$Port")
    if ($IdentityFile) {
        $args += @("-i", $IdentityFile)
    }
    $args += $SourcePath
    $args += $DestinationPath
    & scp @args
    if ($LASTEXITCODE -ne 0) {
        throw "SCP 下载失败: $SourcePath -> $DestinationPath"
    }
}

Ensure-Dir -PathText $LocalOutputParent
$localParent = (Resolve-Path -LiteralPath $LocalOutputParent).Path
$localArchive = Join-Path $localParent "$ResultName.tar.gz"
$localExtractDir = Join-Path $localParent $ResultName

if (Test-Path -LiteralPath $localArchive) {
    Remove-Item -LiteralPath $localArchive -Force
}
if (Test-Path -LiteralPath $localExtractDir) {
    Remove-Item -LiteralPath $localExtractDir -Recurse -Force
}

$remoteArchive = "${RemoteUser}@${RemoteHost}:${RemoteRoot}/${RemoteArchiveRelative}"
Invoke-Scp -SourcePath $remoteArchive -DestinationPath $localArchive

Push-Location $localParent
try {
    & tar -xzf $localArchive
    if ($LASTEXITCODE -ne 0) {
        throw "本地解压结果失败: $localArchive"
    }
}
finally {
    Pop-Location
}

Write-Host "下载完成: $localExtractDir"
Write-Host "GPU效率: $(Join-Path $localExtractDir 'model_efficiency_gpu\model_efficiency_summary.json')"
Write-Host "增强数量: $(Join-Path $localExtractDir 'augmentation_volume\augmentation_volume_report.json')"
Write-Host "Seed重复实验: $(Join-Path $localExtractDir 'seed_repeats\seed_repeat_summary\seed_repeat_summary.csv')"

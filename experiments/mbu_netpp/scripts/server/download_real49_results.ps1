param(
    [Parameter(Mandatory = $true)]
    [string]$RemoteHost,
    [string]$RemoteUser = "ubuntu",
    [int]$Port = 22,
    [string]$IdentityFile = "",
    [string]$RemoteRoot = "/home/ubuntu/wd_bishe_real49",
    [string]$RemoteArchiveRelative = "results/server_delivery/real49_final_infer_100.tar.gz",
    [string]$LocalOutputParent = "D:\中期\wd_bishe\results",
    [string]$ResultName = "real49_final_infer_100"
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
Write-Host "全量推理目录: $(Join-Path $localExtractDir 'infer_all_100')"
Write-Host "剔除训练样本目录: $(Join-Path $localExtractDir 'infer_excluding_train')"

param(
    [Parameter(Mandatory = $true)]
    [string]$RemoteHost,
    [string]$RemoteUser = "root",
    [int]$Port = 22,
    [string]$IdentityFile = "",
    [string]$RemoteRoot = "/root/wd_bishe_real53_suite",
    [string]$LocalRepoRoot = "D:\中期\wd_bishe",
    [string]$LocalTrainSource = "C:\Users\pyd111\Desktop\analysis_same_teacher_nocap_095\1",
    [string]$LocalMicronetCheckpoint = "D:\中期\wd_bishe\experiments\mbu_netpp\workdir\weights\se_resnext50_32x4d_pretrained_microscopynet_v1.0.pth.tar",
    [string]$LocalInferenceInput = "",
    [string]$BundleName = "real53_suite_bundle.tar.gz",
    [switch]$ResetRemote
)

$ErrorActionPreference = "Stop"

function Get-ResolvedPath {
    param([Parameter(Mandatory = $true)][string]$PathText)
    return (Resolve-Path -LiteralPath $PathText).Path
}

function Ensure-Dir {
    param([Parameter(Mandatory = $true)][string]$PathText)
    if (-not (Test-Path -LiteralPath $PathText)) {
        New-Item -ItemType Directory -Path $PathText | Out-Null
    }
}

function Copy-IntoBundle {
    param(
        [Parameter(Mandatory = $true)][string]$SourcePath,
        [Parameter(Mandatory = $true)][string]$DestinationRelative
    )

    $resolvedSource = Get-ResolvedPath -PathText $SourcePath
    $destinationPath = Join-Path $bundleRoot $DestinationRelative
    $parent = Split-Path -Parent $destinationPath
    if ($parent) {
        Ensure-Dir -PathText $parent
    }
    if (Test-Path -LiteralPath $destinationPath) {
        Remove-Item -LiteralPath $destinationPath -Recurse -Force
    }
    Copy-Item -LiteralPath $resolvedSource -Destination $destinationPath -Recurse -Force
}

function Invoke-Ssh {
    param([Parameter(Mandatory = $true)][string]$CommandText)

    $args = @("-p", "$Port")
    if ($IdentityFile) {
        $args += @("-i", $IdentityFile)
    }
    $args += "$RemoteUser@$RemoteHost"
    $args += $CommandText
    & ssh @args
    if ($LASTEXITCODE -ne 0) {
        throw "远端命令执行失败: $CommandText"
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
        throw "SCP 传输失败: $SourcePath -> $DestinationPath"
    }
}

$resolvedRepoRoot = Get-ResolvedPath -PathText $LocalRepoRoot
$tmpRoot = Join-Path $resolvedRepoRoot "tmp\server_bundle_build_real53"
$bundleRoot = Join-Path $tmpRoot "bundle_root"
$archivePath = Join-Path $tmpRoot $BundleName

if (Test-Path -LiteralPath $tmpRoot) {
    Remove-Item -LiteralPath $tmpRoot -Recurse -Force
}
Ensure-Dir -PathText $bundleRoot

Copy-IntoBundle -SourcePath (Join-Path $resolvedRepoRoot "experiments\__init__.py") -DestinationRelative "experiments\__init__.py"

$mbuNetppSourceRoot = Join-Path $resolvedRepoRoot "experiments\mbu_netpp"
$mbuNetppTargetRoot = "experiments\mbu_netpp"
Ensure-Dir -PathText (Join-Path $bundleRoot $mbuNetppTargetRoot)
Get-ChildItem -LiteralPath $mbuNetppSourceRoot -Force |
    Where-Object { $_.Name -notin @("outputs", "workdir", "__pycache__") } |
    ForEach-Object {
        Copy-IntoBundle -SourcePath $_.FullName -DestinationRelative (Join-Path $mbuNetppTargetRoot $_.Name)
    }

Copy-IntoBundle -SourcePath (Join-Path $resolvedRepoRoot "backend\app") -DestinationRelative "backend\app"
Copy-IntoBundle -SourcePath $LocalTrainSource -DestinationRelative "server_assets\training_sources\real53"
Copy-IntoBundle -SourcePath $LocalMicronetCheckpoint -DestinationRelative "experiments\mbu_netpp\workdir\weights\se_resnext50_32x4d_pretrained_microscopynet_v1.0.pth.tar"

if ($LocalInferenceInput -and (Test-Path -LiteralPath $LocalInferenceInput)) {
    Copy-IntoBundle -SourcePath $LocalInferenceInput -DestinationRelative "server_assets\final_infer_inputs\full_png_cropped_xlsx_images"
}

Push-Location $bundleRoot
try {
    if (Test-Path -LiteralPath $archivePath) {
        Remove-Item -LiteralPath $archivePath -Force
    }
    & tar -czf $archivePath .
    if ($LASTEXITCODE -ne 0) {
        throw "打包 bundle 失败"
    }
}
finally {
    Pop-Location
}

if ($ResetRemote) {
    Invoke-Ssh -CommandText "rm -rf '$RemoteRoot' && mkdir -p '$RemoteRoot/incoming'"
}
else {
    Invoke-Ssh -CommandText "mkdir -p '$RemoteRoot/incoming'"
}

$remoteArchive = "${RemoteUser}@${RemoteHost}:${RemoteRoot}/incoming/${BundleName}"
Invoke-Scp -SourcePath $archivePath -DestinationPath $remoteArchive
Invoke-Ssh -CommandText "mkdir -p '$RemoteRoot' && tar -xzf '$RemoteRoot/incoming/$BundleName' -C '$RemoteRoot'"

Write-Host "上传完成"
Write-Host "远端根目录: $RemoteRoot"
Write-Host "远端 bundle: $RemoteRoot/incoming/$BundleName"
Write-Host "建议下一步先检查环境："
Write-Host "  ssh -p $Port $RemoteUser@$RemoteHost"
Write-Host "  cd $RemoteRoot"
Write-Host "  bash experiments/mbu_netpp/scripts/server/run_real53_experiments.sh"

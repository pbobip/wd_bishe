param(
    [string]$ProjectRoot = "C:\Users\pyd111\Desktop\中期\wd_bishe",
    [string]$AiFRoot = "D:\BaiduNetdiskDownload\AI-F",
    [string]$BilibiliRoot = "D:\BaiduNetdiskDownload\B站更新（和视频同步更新 日更）\B站更新（和视频同步更新 日更）"
)

$ErrorActionPreference = "Stop"

function Get-RelativePathCompat {
    param(
        [Parameter(Mandatory = $true)][string]$BasePath,
        [Parameter(Mandatory = $true)][string]$TargetPath
    )

    $baseFull = [System.IO.Path]::GetFullPath($BasePath)
    if (-not $baseFull.EndsWith([System.IO.Path]::DirectorySeparatorChar)) {
        $baseFull += [System.IO.Path]::DirectorySeparatorChar
    }
    $targetFull = [System.IO.Path]::GetFullPath($TargetPath)
    $baseUri = New-Object System.Uri($baseFull)
    $targetUri = New-Object System.Uri($targetFull)
    $relativeUri = $baseUri.MakeRelativeUri($targetUri)
    return [System.Uri]::UnescapeDataString($relativeUri.ToString()).Replace('/', [System.IO.Path]::DirectorySeparatorChar)
}

$destRoot = Join-Path $ProjectRoot "tools\module_reference_sync"
$sourceDestRoot = Join-Path $destRoot "sources"
$inventoryPath = Join-Path $destRoot "inventory.csv"
$duplicatePath = Join-Path $destRoot "duplicates_by_sha256.csv"
$summaryPath = Join-Path $destRoot "summary.json"

New-Item -ItemType Directory -Force -Path $destRoot | Out-Null
if (Test-Path $sourceDestRoot) {
    Remove-Item -LiteralPath $sourceDestRoot -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $sourceDestRoot | Out-Null

$sources = @(
    @{
        Name = "AI-F"
        Root = $AiFRoot
        Dest = "ai_f_py"
    },
    @{
        Name = "B站更新（和视频同步更新 日更）"
        Root = $BilibiliRoot
        Dest = "bilibili_daily_py"
    }
)

$inventory = New-Object System.Collections.Generic.List[object]
$summary = New-Object System.Collections.Generic.List[object]

foreach ($source in $sources) {
    if (-not (Test-Path -LiteralPath $source.Root)) {
        throw "源目录不存在: $($source.Root)"
    }

    $targetRoot = Join-Path $sourceDestRoot $source.Dest
    New-Item -ItemType Directory -Force -Path $targetRoot | Out-Null

    $files = Get-ChildItem -LiteralPath $source.Root -Recurse -File -Filter *.py | Sort-Object FullName
    $totalBytes = 0L

    foreach ($file in $files) {
        $relative = Get-RelativePathCompat -BasePath $source.Root -TargetPath $file.FullName
        $destPath = Join-Path $targetRoot $relative
        $destDir = Split-Path -Parent $destPath
        New-Item -ItemType Directory -Force -Path $destDir | Out-Null
        Copy-Item -LiteralPath $file.FullName -Destination $destPath -Force

        $hash = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash
        $size = [int64]$file.Length
        $totalBytes += $size

        $inventory.Add([pscustomobject]@{
            library = $source.Name
            source_root = $source.Root
            source_relative_path = $relative
            destination_relative_path = Get-RelativePathCompat -BasePath $ProjectRoot -TargetPath $destPath
            size_bytes = $size
            sha256 = $hash
        }) | Out-Null
    }

    $summary.Add([pscustomobject]@{
        library = $source.Name
        source_root = $source.Root
        destination_root = Get-RelativePathCompat -BasePath $ProjectRoot -TargetPath $targetRoot
        file_count = $files.Count
        total_size_bytes = $totalBytes
    }) | Out-Null
}

$inventory | Export-Csv -LiteralPath $inventoryPath -NoTypeInformation -Encoding UTF8

$duplicates = $inventory |
    Group-Object sha256 |
    Where-Object { $_.Count -gt 1 } |
    Sort-Object Count -Descending |
    ForEach-Object {
        [pscustomobject]@{
            sha256 = $_.Name
            duplicate_count = $_.Count
            libraries = (($_.Group.library | Sort-Object -Unique) -join "; ")
            source_relative_paths = (($_.Group | ForEach-Object { "$($_.library):$($_.source_relative_path)" }) -join " | ")
            destination_relative_paths = (($_.Group.destination_relative_path) -join " | ")
        }
    }

$duplicates | Export-Csv -LiteralPath $duplicatePath -NoTypeInformation -Encoding UTF8

$summaryPayload = [pscustomobject]@{
    generated_at = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss K")
    project_root = $ProjectRoot
    total_python_files = $inventory.Count
    total_python_size_bytes = ($inventory | Measure-Object size_bytes -Sum).Sum
    duplicate_sha256_groups = $duplicates.Count
    duplicate_file_entries = ($duplicates | Measure-Object duplicate_count -Sum).Sum
    libraries = $summary
}

$summaryPayload | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $summaryPath -Encoding UTF8

Write-Output "模块源码整理完成: $destRoot"
Write-Output "Python 文件总数: $($inventory.Count)"
Write-Output "重复哈希组数: $($duplicates.Count)"

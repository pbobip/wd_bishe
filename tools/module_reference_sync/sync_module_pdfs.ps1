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

function Get-PdfCategory {
    param([string]$RelativePath)

    $text = $RelativePath.ToLowerInvariant()
    $guideKeywords = @(
        "指南", "指导", "教程", "答疑", "分析", "创新", "简介", "介绍", "制作", "开篇", "思维",
        "查找", "案例", "基础", "进阶", "例子", "prompt", "目录", "架构", "如何", "效果", "写作"
    )

    foreach ($keyword in $guideKeywords) {
        if ($text.Contains($keyword.ToLowerInvariant())) {
            return "guides"
        }
    }

    return "papers"
}

function Get-SafePdfFileName {
    param(
        [Parameter(Mandatory = $true)][string]$RelativePath,
        [Parameter(Mandatory = $true)][string]$Hash
    )

    $baseName = [System.IO.Path]::GetFileNameWithoutExtension($RelativePath)
    $invalid = [System.IO.Path]::GetInvalidFileNameChars() + @([char]':')
    foreach ($char in $invalid) {
        $baseName = $baseName.Replace($char, '_')
    }
    $baseName = ($baseName -replace '\s+', '_').Trim('_')
    if ([string]::IsNullOrWhiteSpace($baseName)) {
        $baseName = "document"
    }
    if ($baseName.Length -gt 80) {
        $baseName = $baseName.Substring(0, 80)
    }
    return "{0}__{1}.pdf" -f $baseName, $Hash.Substring(0, 8)
}

$destRoot = Join-Path $ProjectRoot "tools\module_reference_sync"
$pdfDestRoot = Join-Path $destRoot "pdfs"
$inventoryPath = Join-Path $destRoot "pdf_inventory.csv"
$duplicatePath = Join-Path $destRoot "pdf_duplicates_by_sha256.csv"
$summaryPath = Join-Path $destRoot "pdf_summary.json"

New-Item -ItemType Directory -Force -Path $destRoot | Out-Null
if (Test-Path $pdfDestRoot) {
    Remove-Item -LiteralPath $pdfDestRoot -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $pdfDestRoot | Out-Null

$sources = @(
    @{
        Name = "AI-F"
        Root = $AiFRoot
        Dest = "ai_f"
    },
    @{
        Name = "B站更新（和视频同步更新 日更）"
        Root = $BilibiliRoot
        Dest = "bilibili_daily"
    }
)

$rows = New-Object System.Collections.Generic.List[object]

foreach ($source in $sources) {
    if (-not (Test-Path -LiteralPath $source.Root)) {
        throw "源目录不存在: $($source.Root)"
    }

    $files = Get-ChildItem -LiteralPath $source.Root -Recurse -File -Filter *.pdf | Sort-Object FullName
    foreach ($file in $files) {
        $relative = Get-RelativePathCompat -BasePath $source.Root -TargetPath $file.FullName
        $hash = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash
        $category = Get-PdfCategory -RelativePath $relative
        $rows.Add([pscustomobject]@{
            library = $source.Name
            library_dest = $source.Dest
            source_root = $source.Root
            source_full_path = $file.FullName
            source_relative_path = $relative
            size_bytes = [int64]$file.Length
            sha256 = $hash
            category = $category
        }) | Out-Null
    }
}

$canonicalByHash = @{}
foreach ($group in ($rows | Group-Object sha256)) {
    $canonical = $group.Group |
        Sort-Object @{Expression = { $_.source_relative_path.Length }}, @{Expression = { $_.source_relative_path }} |
        Select-Object -First 1
    $canonicalByHash[$group.Name] = $canonical
}

$inventory = New-Object System.Collections.Generic.List[object]
$duplicates = New-Object System.Collections.Generic.List[object]

foreach ($group in ($rows | Group-Object sha256 | Sort-Object @{ Expression = "Count"; Descending = $true }, @{ Expression = "Name"; Descending = $false })) {
    $canonical = $canonicalByHash[$group.Name]
    $destFileName = Get-SafePdfFileName -RelativePath $canonical.source_relative_path -Hash $group.Name
    $canonicalDest = Join-Path $pdfDestRoot (Join-Path $canonical.category (Join-Path $canonical.library_dest $destFileName))
    $canonicalDir = Split-Path -Parent $canonicalDest
    New-Item -ItemType Directory -Force -Path $canonicalDir | Out-Null
    if (-not (Test-Path -LiteralPath $canonicalDest)) {
        Copy-Item -LiteralPath $canonical.source_full_path -Destination $canonicalDest -Force
    }

    if ($group.Count -gt 1) {
        $duplicates.Add([pscustomobject]@{
            sha256 = $group.Name
            duplicate_count = $group.Count
            canonical_source = "$($canonical.library):$($canonical.source_relative_path)"
            canonical_destination_relative_path = Get-RelativePathCompat -BasePath $ProjectRoot -TargetPath $canonicalDest
            duplicate_sources = ($group.Group | ForEach-Object { "$($_.library):$($_.source_relative_path)" }) -join " | "
        }) | Out-Null
    }

    foreach ($item in $group.Group) {
        $inventory.Add([pscustomobject]@{
            library = $item.library
            category = $item.category
            source_root = $item.source_root
            source_relative_path = $item.source_relative_path
            size_bytes = $item.size_bytes
            sha256 = $item.sha256
            is_canonical = [bool]($item.source_full_path -eq $canonical.source_full_path)
            canonical_destination_relative_path = Get-RelativePathCompat -BasePath $ProjectRoot -TargetPath $canonicalDest
        }) | Out-Null
    }
}

$inventory | Export-Csv -LiteralPath $inventoryPath -NoTypeInformation -Encoding UTF8
$duplicates | Export-Csv -LiteralPath $duplicatePath -NoTypeInformation -Encoding UTF8

$uploadedCanonical = $inventory | Where-Object { $_.is_canonical -eq $true }
$summaryPayload = [pscustomobject]@{
    generated_at = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss K")
    project_root = $ProjectRoot
    total_pdf_entries = $inventory.Count
    uploaded_canonical_pdf_count = $uploadedCanonical.Count
    uploaded_canonical_size_bytes = ($uploadedCanonical | Measure-Object size_bytes -Sum).Sum
    duplicate_sha256_groups = $duplicates.Count
    duplicate_file_entries = (($duplicates | Measure-Object duplicate_count -Sum).Sum)
    categories = @(
        [pscustomobject]@{
            category = "papers"
            canonical_count = (@($uploadedCanonical | Where-Object { $_.category -eq "papers" }).Count)
            canonical_size_bytes = (($uploadedCanonical | Where-Object { $_.category -eq "papers" } | Measure-Object size_bytes -Sum).Sum)
        },
        [pscustomobject]@{
            category = "guides"
            canonical_count = (@($uploadedCanonical | Where-Object { $_.category -eq "guides" }).Count)
            canonical_size_bytes = (($uploadedCanonical | Where-Object { $_.category -eq "guides" } | Measure-Object size_bytes -Sum).Sum)
        }
    )
    libraries = @(
        [pscustomobject]@{
            library = "AI-F"
            entry_count = (($inventory | Where-Object { $_.library -eq "AI-F" }).Count)
            canonical_count = (@($uploadedCanonical | Where-Object { $_.library -eq "AI-F" }).Count)
            total_size_bytes = (($inventory | Where-Object { $_.library -eq "AI-F" } | Measure-Object size_bytes -Sum).Sum)
        },
        [pscustomobject]@{
            library = "B站更新（和视频同步更新 日更）"
            entry_count = (($inventory | Where-Object { $_.library -eq "B站更新（和视频同步更新 日更）" }).Count)
            canonical_count = (@($uploadedCanonical | Where-Object { $_.library -eq "B站更新（和视频同步更新 日更）" }).Count)
            total_size_bytes = (($inventory | Where-Object { $_.library -eq "B站更新（和视频同步更新 日更）" } | Measure-Object size_bytes -Sum).Sum)
        }
    )
}

$summaryPayload | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $summaryPath -Encoding UTF8

Write-Output "PDF 整理完成: $pdfDestRoot"
Write-Output "PDF 条目总数: $($inventory.Count)"
Write-Output "去重后上传数: $($uploadedCanonical.Count)"
Write-Output "重复哈希组数: $($duplicates.Count)"

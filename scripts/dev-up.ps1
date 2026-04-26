[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
$frontendDir = Join-Path $repoRoot 'frontend'
$backendDir = Join-Path $repoRoot 'backend'
$outputDir = Join-Path $repoRoot 'output'
$frontendPort = 5173
$backendPort = 8000

function Write-Info([string]$message) {
  Write-Host "[wd_bishe] $message"
}

function Resolve-PythonExe {
  $preferred = Join-Path $env:USERPROFILE 'fy\Scripts\python.exe'
  if (Test-Path $preferred) {
    return $preferred
  }

  $python = Get-Command python -ErrorAction SilentlyContinue
  if ($python) {
    return $python.Source
  }

  throw '未找到 Python。请先确认虚拟环境或系统 Python 可用。'
}

function Resolve-NpmExe {
  $npmCmd = Get-Command npm.cmd -ErrorAction SilentlyContinue
  if ($npmCmd) {
    return $npmCmd.Source
  }

  $npm = Get-Command npm -ErrorAction SilentlyContinue
  if ($npm) {
    return $npm.Source
  }

  throw '未找到 npm。请先安装 Node.js，并确保 npm 在 PATH 中。'
}

function Contains-IgnoreCase([string]$text, [string]$fragment) {
  if ([string]::IsNullOrWhiteSpace($text)) {
    return $false
  }

  return $text.IndexOf($fragment, [System.StringComparison]::OrdinalIgnoreCase) -ge 0
}

function Stop-RepoProcesses {
  param(
    [string]$label,
    [scriptblock]$predicate
  )

  $targets = Get-CimInstance Win32_Process | Where-Object { & $predicate $_ }
  foreach ($process in $targets) {
    Write-Info ("结束已有{0}进程 PID={1}" -f $label, $process.ProcessId)
    Stop-Process -Id $process.ProcessId -Force -ErrorAction SilentlyContinue
  }
}

function Get-ListeningProcessInfo([int]$port) {
  $listener = Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue | Select-Object -First 1
  if (-not $listener) {
    return $null
  }

  $process = Get-CimInstance Win32_Process -Filter "ProcessId = $($listener.OwningProcess)" -ErrorAction SilentlyContinue
  if (-not $process) {
    return [pscustomobject]@{
      ProcessId = $listener.OwningProcess
      CommandLine = ''
      Name = ''
    }
  }

  return [pscustomobject]@{
    ProcessId = $process.ProcessId
    CommandLine = $process.CommandLine
    Name = $process.Name
  }
}

function Test-ListenerMatchesCurrentRepo {
  param(
    [int]$port,
    [ValidateSet('frontend', 'backend')]
    [string]$kind
  )

  $listener = Get-ListeningProcessInfo -port $port
  if (-not $listener) {
    return $false
  }

  if ($kind -eq 'frontend') {
    return
      (Contains-IgnoreCase $listener.CommandLine $frontendDir) -and
      (Contains-IgnoreCase $listener.CommandLine 'vite')
  }

  return
    (Contains-IgnoreCase $listener.CommandLine $repoRoot) -and
    (Contains-IgnoreCase $listener.CommandLine 'backend.main:app') -and
    (Contains-IgnoreCase $listener.CommandLine '--reload')
}

function Assert-PortAvailable([int]$port, [string]$role) {
  $listener = Get-ListeningProcessInfo -port $port
  if (-not $listener) {
    return
  }

  $details = if ($listener.CommandLine) { $listener.CommandLine } else { $listener.Name }
  throw ("{0} 端口 {1} 已被其他进程占用。PID={2}，命令={3}" -f $role, $port, $listener.ProcessId, $details)
}

function Wait-PortReleased {
  param(
    [int]$port,
    [int]$timeoutSeconds = 12
  )

  $deadline = (Get-Date).AddSeconds($timeoutSeconds)
  while ((Get-Date) -lt $deadline) {
    $listener = Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $listener) {
      return $true
    }

    Start-Sleep -Milliseconds 500
  }

  return $false
}

function Stop-PortListenerIfManaged {
  param(
    [int]$port,
    [string]$role
  )

  $listener = Get-ListeningProcessInfo -port $port
  if (-not $listener) {
    return
  }

  $managedByRepo = $false
  if ($listener.CommandLine) {
    $managedByRepo =
      (Contains-IgnoreCase $listener.CommandLine $repoRoot) -or
      (Contains-IgnoreCase $listener.CommandLine 'wd_bishe_backend') -or
      (Contains-IgnoreCase $listener.CommandLine 'wd_bishe_frontend') -or
      ((Contains-IgnoreCase $listener.CommandLine 'wd_bishe\frontend') -and (Contains-IgnoreCase $listener.CommandLine 'vite')) -or
      (Contains-IgnoreCase $listener.CommandLine 'backend.main:app') -or
      (Contains-IgnoreCase $listener.CommandLine 'vite')
  }

  if (-not $managedByRepo -and -not [string]::IsNullOrWhiteSpace($listener.CommandLine)) {
    return
  }

  Write-Info ("结束残留{0}监听 PID={1}" -f $role, $listener.ProcessId)
  & taskkill /PID $listener.ProcessId /T /F | Out-Null
}

function Wait-HttpReady {
  param(
    [string]$uri,
    [int]$timeoutSeconds
  )

  $deadline = (Get-Date).AddSeconds($timeoutSeconds)
  while ((Get-Date) -lt $deadline) {
    try {
      $response = Invoke-WebRequest -UseBasicParsing -Uri $uri -TimeoutSec 2
      if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 500) {
        return $true
      }
    } catch {
      Start-Sleep -Milliseconds 600
    }
  }

  return $false
}

$pythonExe = Resolve-PythonExe
$npmExe = Resolve-NpmExe

if (-not (Test-Path $outputDir)) {
  New-Item -ItemType Directory -Path $outputDir | Out-Null
}

$backendLog = Join-Path $outputDir 'backend-dev.log'
$frontendLog = Join-Path $outputDir 'frontend-dev.log'

$backendCommand = @'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$Host.UI.RawUI.WindowTitle = 'wd_bishe_backend'
Set-Location '{0}'
Start-Transcript -Path '{1}' -Append | Out-Null
try {{
  & '{2}' -m uvicorn backend.main:app --host 127.0.0.1 --port {3} --app-dir '{0}' --reload --reload-dir '{0}\backend'
}} finally {{
  Stop-Transcript | Out-Null
}}
'@ -f $repoRoot, $backendLog, $pythonExe, $backendPort

$frontendCommand = @'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$Host.UI.RawUI.WindowTitle = 'wd_bishe_frontend'
Set-Location '{0}'
Start-Transcript -Path '{1}' -Append | Out-Null
try {{
  & '{2}' run dev
}} finally {{
  Stop-Transcript | Out-Null
}}
'@ -f $frontendDir, $frontendLog, $npmExe

$frontendProcess = $null
$backendProcess = $null
$frontendReady =
  (Test-ListenerMatchesCurrentRepo -port $frontendPort -kind 'frontend') -and
  (Wait-HttpReady -uri "http://127.0.0.1:$frontendPort/" -timeoutSeconds 2)
$backendReady =
  (Test-ListenerMatchesCurrentRepo -port $backendPort -kind 'backend') -and
  (Wait-HttpReady -uri "http://127.0.0.1:$backendPort/api/health" -timeoutSeconds 2)

if (-not $frontendReady) {
  Stop-RepoProcesses -label '前端' -predicate {
    param($process)
    $process.Name -eq 'node.exe' -and
    (Contains-IgnoreCase $process.CommandLine $frontendDir) -and
    ((Contains-IgnoreCase $process.CommandLine 'vite') -or (Contains-IgnoreCase $process.CommandLine 'npm-cli.js run dev'))
  }

  Stop-RepoProcesses -label '前端启动窗口' -predicate {
    param($process)
    $process.Name -eq 'powershell.exe' -and (Contains-IgnoreCase $process.CommandLine 'wd_bishe_frontend')
  }

  Start-Sleep -Milliseconds 800
  Stop-PortListenerIfManaged -port $frontendPort -role '前端'
  if (-not (Wait-PortReleased -port $frontendPort)) {
    Assert-PortAvailable -port $frontendPort -role '前端'
  }
  Assert-PortAvailable -port $frontendPort -role '前端'

  if (Test-Path $frontendLog) {
    Remove-Item -LiteralPath $frontendLog -Force -ErrorAction SilentlyContinue
  }

  Write-Info '启动前端窗口...'
  $frontendProcess = Start-Process -FilePath 'powershell.exe' -ArgumentList @('-NoExit', '-Command', $frontendCommand) -WorkingDirectory $frontendDir -PassThru
  $frontendReady = Wait-HttpReady -uri "http://127.0.0.1:$frontendPort/" -timeoutSeconds 18
} else {
  Write-Info '前端已在运行，复用现有实例。'
}

if (-not $backendReady) {
  Stop-RepoProcesses -label '后端' -predicate {
    param($process)
    $process.Name -eq 'python.exe' -and
    (Contains-IgnoreCase $process.CommandLine $repoRoot) -and
    (Contains-IgnoreCase $process.CommandLine 'uvicorn') -and
    (Contains-IgnoreCase $process.CommandLine 'backend.main:app')
  }

  Stop-RepoProcesses -label '后端启动窗口' -predicate {
    param($process)
    $process.Name -eq 'powershell.exe' -and (Contains-IgnoreCase $process.CommandLine 'wd_bishe_backend')
  }

  Start-Sleep -Milliseconds 800
  Stop-PortListenerIfManaged -port $backendPort -role '后端'
  if (-not (Wait-PortReleased -port $backendPort)) {
    if (Wait-HttpReady -uri "http://127.0.0.1:$backendPort/api/health" -timeoutSeconds 2) {
      $backendReady = $true
      Write-Info '后端已在运行，复用现有实例。'
    } else {
      Assert-PortAvailable -port $backendPort -role '后端'
    }
  }

  if (-not $backendReady) {
    Assert-PortAvailable -port $backendPort -role '后端'
    if (Test-Path $backendLog) {
      Remove-Item -LiteralPath $backendLog -Force -ErrorAction SilentlyContinue
    }

    Write-Info '启动后端窗口...'
    $backendProcess = Start-Process -FilePath 'powershell.exe' -ArgumentList @('-NoExit', '-Command', $backendCommand) -WorkingDirectory $repoRoot -PassThru
    $backendReady = Wait-HttpReady -uri "http://127.0.0.1:$backendPort/api/health" -timeoutSeconds 18
  }
} else {
  Write-Info '后端已在运行，复用现有实例。'
}

if (-not $backendReady) {
  Write-Warning "后端未在预期时间内就绪，请检查 $backendLog 或后端窗口。"
}

if (-not $frontendReady) {
  Write-Warning "前端未在预期时间内就绪，请检查 $frontendLog 或前端窗口。"
}

$frontendPidLabel = if ($frontendProcess) { $frontendProcess.Id } else { '沿用现有实例' }
$backendPidLabel = if ($backendProcess) { $backendProcess.Id } else { '沿用现有实例' }

Write-Info ("前端地址: http://127.0.0.1:{0}/" -f $frontendPort)
Write-Info ("后端地址: http://127.0.0.1:{0}/api/health" -f $backendPort)
Write-Info ("前端窗口 PID={0}，后端窗口 PID={1}" -f $frontendPidLabel, $backendPidLabel)
Write-Info "日志目录: $outputDir"

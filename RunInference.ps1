param(
    [string]$Config = "Automation\automation_config.yaml",
    [string]$Command = "",
    [string]$LogDirectory = "logs"
)

$ProjectPython = Join-Path $PSScriptRoot "RL_venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $ProjectPython)) {
    $ProjectPython = "python"
}

if (-not [System.IO.Path]::IsPathRooted($LogDirectory)) {
    $LogDirectory = Join-Path $PSScriptRoot $LogDirectory
}
New-Item -ItemType Directory -Path $LogDirectory -Force -ErrorAction Stop | Out-Null

$RunTimestamp = Get-Date -Format "yyyyMMdd_HHmmss_fff"
$LogPath = Join-Path $LogDirectory "inference_$RunTimestamp.txt"
$PythonArguments = @("-m", "Automation.infer", "--config", $Config)
if ($Command) {
    $PythonArguments += @("--command", $Command)
}

function Write-RunLog {
    param([string]$Text)

    Write-Host $Text
    $Text | Out-File -LiteralPath $LogPath -Append -Encoding utf8
}

@(
    "RL inference run"
    "Started: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')"
    "Working directory: $PSScriptRoot"
    "Config: $Config"
    "Command: $ProjectPython $($PythonArguments -join ' ')"
    ("=" * 80)
) | ForEach-Object { Write-RunLog $_ }

$ProcessExitCode = 1
try {
    & $ProjectPython @PythonArguments 2>&1 | ForEach-Object {
        Write-RunLog $_.ToString()
    }
    $ProcessExitCode = $LASTEXITCODE
    if ($null -eq $ProcessExitCode) {
        $ProcessExitCode = 0
    }
} catch {
    Write-RunLog "PowerShell launcher error:"
    Write-RunLog ($_ | Out-String).TrimEnd()
    $ProcessExitCode = 1
}

Write-RunLog ("=" * 80)
Write-RunLog "Finished: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')"
Write-RunLog "Exit code: $ProcessExitCode"
Write-RunLog "Log file: $LogPath"

if ($ProcessExitCode -ne 0) {
    throw "RL inference failed with exit code $ProcessExitCode. See $LogPath"
}

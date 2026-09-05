param(
    [ValidateSet("Train", "Dataset", "Live")]
    [string]$Mode = "Train",
    [ValidateSet("BC", "AWAC")]
    [string]$Algorithm = "BC",
    [string]$CheckpointPath = "",
    [string]$DatasetId = "env16/BC-v0",
    [int]$EvalEpisodes = 5,
    [int]$UpdateSteps = 1000000,
    [int]$OnlineIterations = 0,
    [int]$HiddenDim = 256,
    [double]$LearningRate = 0.0003,
    [double]$Tau = 0.005,
    [double]$AwacLambda = 1.0,
    [int]$BufferSize = 2000000,
    [int]$BatchSize = 256,
    [int]$EvalEvery = 5000,
    [double]$TopFraction = 1.0,
    [double]$Gamma = 0.99,
    [ValidateSet("auto", "cpu", "cuda")]
    [string]$Device = "auto",
    [switch]$NoNormalizeState,
    [string]$CheckpointsPath = "runs",
    [string]$OutputCsv = "",
    [string]$Config = "Automation\automation_config.yaml",
    [string]$LogDirectory = "logs"
)

$ProjectPython = Join-Path $PSScriptRoot "RL_venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $ProjectPython)) {
    $ProjectPython = "python"
}

if (-not [System.IO.Path]::IsPathRooted($LogDirectory)) {
    $LogDirectory = Join-Path $PSScriptRoot $LogDirectory
}
New-Item -ItemType Directory -Path $LogDirectory -Force -ErrorAction Stop |
    Out-Null

$RunTimestamp = Get-Date -Format "yyyyMMdd_HHmmss_fff"
$LogPath = Join-Path $LogDirectory "offline_rl_$RunTimestamp.txt"
$PythonArguments = @(
    "-m", "Automation.offline_rl",
    "--config", $Config,
    "--mode", $Mode.ToLowerInvariant(),
    "--algorithm", $Algorithm.ToLowerInvariant(),
    "--dataset-id", $DatasetId,
    "--eval-episodes", $EvalEpisodes,
    "--update-steps", $UpdateSteps,
    "--online-iterations", $OnlineIterations,
    "--hidden-dim", $HiddenDim,
    "--learning-rate", $LearningRate,
    "--tau", $Tau,
    "--awac-lambda", $AwacLambda,
    "--buffer-size", $BufferSize,
    "--batch-size", $BatchSize,
    "--eval-every", $EvalEvery,
    "--checkpoints-path", $CheckpointsPath,
    "--top-fraction", $TopFraction,
    "--gamma", $Gamma,
    "--device", $Device
)
if ($CheckpointPath) {
    $PythonArguments += @("--checkpoint-path", $CheckpointPath)
}
if ($NoNormalizeState) {
    $PythonArguments += "--no-normalize-state"
} else {
    $PythonArguments += "--normalize-state"
}
if ($OutputCsv) {
    $PythonArguments += @("--output-csv", $OutputCsv)
}

function Write-RunLog {
    param([string]$Text)

    Write-Host $Text
}

$TranscriptStarted = $false
try {
    Start-Transcript -LiteralPath $LogPath -Append -ErrorAction Stop | Out-Null
    $TranscriptStarted = $true
} catch {
    Write-Warning "Could not start transcript at ${LogPath}: $($_.Exception.Message)"
}

@(
    "Offline RL $Algorithm"
    "Started: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')"
    "Working directory: $PSScriptRoot"
    "Mode: $Mode"
    "Dataset: $DatasetId"
    "Checkpoint: $CheckpointPath"
    "Command: $ProjectPython $($PythonArguments -join ' ')"
    ("=" * 80)
) | ForEach-Object { Write-RunLog $_ }

$ProcessExitCode = 1
try {
    & $ProjectPython @PythonArguments
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

if ($TranscriptStarted) {
    Stop-Transcript | Out-Null
}

if ($ProcessExitCode -ne 0) {
    throw "Offline RL failed with exit code $ProcessExitCode. See $LogPath"
}

param(
    [string]$Config = "Automation\automation_config.yaml",
    [string]$Command = ""
)

$ProjectPython = Join-Path $PSScriptRoot "RL_venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $ProjectPython)) {
    $ProjectPython = "python"
}

if ($Command) {
    & $ProjectPython -m Automation.infer --config $Config --command $Command
} else {
    & $ProjectPython -m Automation.infer --config $Config
}

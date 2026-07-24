param(
    [string]$Config = "Automation\automation_config.yaml"
)

$ProjectPython = Join-Path $PSScriptRoot "RL_venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $ProjectPython)) {
    $ProjectPython = "python"
}

& $ProjectPython -m Automation.train --config $Config

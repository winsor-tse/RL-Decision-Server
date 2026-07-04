param(
    [string]$Config = "Automation\automation_config.yaml"
)

python Automation\Training_stack.py --config $Config

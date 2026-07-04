param(
    [string]$Config = "Automation\automation_config.yaml",
    [string]$Command = ""
)

if ($Command) {
    python Automation\Inference_stack.py --config $Config --command $Command
} else {
    python Automation\Inference_stack.py --config $Config
}

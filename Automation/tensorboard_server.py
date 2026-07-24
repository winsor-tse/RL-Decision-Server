"""Run TensorBoard without noisy optional-dependency notices."""

import sys
import warnings


PKG_RESOURCES_WARNING = r"pkg_resources is deprecated as an API\..*"
NO_TENSORFLOW_NOTICE = (
    "TensorFlow installation not found - running with reduced feature set."
)


class FilteredStderr:
    """Forward stderr except for TensorBoard's expected no-TensorFlow notice."""

    def __init__(self, stream):
        self.stream = stream
        self.suppress_next_newline = False

    def write(self, message):
        if NO_TENSORFLOW_NOTICE in message:
            self.suppress_next_newline = not message.endswith(("\n", "\r"))
            return len(message)
        if self.suppress_next_newline and message in {"\n", "\r\n"}:
            self.suppress_next_newline = False
            return len(message)
        self.suppress_next_newline = False
        return self.stream.write(message)

    def flush(self):
        return self.stream.flush()

    def isatty(self):
        return self.stream.isatty()

    def __getattr__(self, name):
        return getattr(self.stream, name)

    @property
    def encoding(self):
        return self.stream.encoding


def main() -> None:
    warnings.filterwarnings(
        "ignore",
        message=PKG_RESOURCES_WARNING,
        category=UserWarning,
        module=r"tensorboard\.default",
    )

    # Import after installing the filter: tensorboard.default imports
    # pkg_resources while the module is initialized.
    from tensorboard import main as tensorboard_main

    original_stderr = sys.stderr
    sys.stderr = FilteredStderr(original_stderr)
    try:
        tensorboard_main.run_main()
    finally:
        sys.stderr = original_stderr


if __name__ == "__main__":
    main()

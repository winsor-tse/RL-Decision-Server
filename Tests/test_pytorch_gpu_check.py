import unittest

from Tests.pytorch_gpu_check import (
    NvidiaGpu,
    NvidiaSystem,
    recommend_cuda_wheel,
    version_tuple,
)


def nvidia_system(
    maximum_cuda: tuple[int, int],
    compute_capability: tuple[int, int] | None,
) -> NvidiaSystem:
    return NvidiaSystem(
        executable="nvidia-smi",
        maximum_cuda=maximum_cuda,
        gpus=(
            NvidiaGpu(
                index=0,
                name="Example NVIDIA GPU",
                driver_version="999.0",
                memory_mib=8192,
                compute_capability=compute_capability,
            ),
        ),
    )


class PyTorchGpuCheckTests(unittest.TestCase):
    def test_version_parser_ignores_build_suffixes(self):
        self.assertEqual(version_tuple("13.0+vendor"), (13, 0))
        self.assertIsNone(version_tuple(None))

    def test_blackwell_and_cuda_13_driver_select_cu130(self):
        recommendation = recommend_cuda_wheel(
            nvidia_system((13, 1), (12, 0)),
            device_index=0,
        )

        self.assertEqual(recommendation, ((13, 0), "cu130"))

    def test_blackwell_and_cuda_12_8_driver_select_cu128(self):
        recommendation = recommend_cuda_wheel(
            nvidia_system((12, 8), (12, 0)),
            device_index=0,
        )

        self.assertEqual(recommendation, ((12, 8), "cu128"))

    def test_blackwell_rejects_cuda_12_6_only_driver(self):
        recommendation = recommend_cuda_wheel(
            nvidia_system((12, 6), (12, 0)),
            device_index=0,
        )

        self.assertIsNone(recommendation)

    def test_unknown_compute_capability_does_not_guess(self):
        recommendation = recommend_cuda_wheel(
            nvidia_system((13, 1), None),
            device_index=0,
        )

        self.assertIsNone(recommendation)


if __name__ == "__main__":
    unittest.main()

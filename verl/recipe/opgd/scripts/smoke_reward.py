"""Deterministic non-semantic reward used only by the OPGD GPU smoke test."""


def compute_score(
    data_source,
    solution_str: str,
    ground_truth,
    extra_info=None,
    **kwargs,
) -> float:
    del data_source, ground_truth, extra_info, kwargs
    encoded_response = solution_str.encode("utf-8", errors="replace")
    checksum = sum(
        (byte_index + 1) * byte_value
        for byte_index, byte_value in enumerate(encoded_response)
    )
    return float(checksum % 1009) / 1009.0

import os
import torch
from development_phase import evaluate
from pypapi import events, papi_high

os.environ["PAPI_EVENTS"] = "PAPI_SP_OPS"  # FLOPs in single precision
os.environ["PAPI_OUTPUT_DIRECTORY"] = "papi_output"


def main():
    network_input = torch.ones([1, 4, 512, 512], dtype=torch.float32)

    papi_high.hl_region_begin("evaluator")
    print(evaluate(network_input))
    papi_high.hl_region_end("evaluator")


if __name__ == "__main__":
    main()

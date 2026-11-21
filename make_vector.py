#!/usr/bin/env python3
"""Write a llama.cpp control vector that pushes one expert neuron of Inkling 2 Mini.

    python make_vector.py Inkling-2-Mini-Q8_0.gguf evil   evil-L11E14N370.gguf
    python make_vector.py Inkling-2-Mini-Q8_0.gguf sexual sexual-L7E78N1060.gguf

The vector is written at dose 1.0. The dose is the scale you pass to llama.cpp:
    --control-vector-scaled evil-L11E14N370.gguf:0.75

Recipe: dose * median(||resid leaving layer L||, tokens 1:) / ||W_down[E][:, n]|| added to neuron n of expert E
at the down_proj input, every token. down_proj is linear, so this is one fixed vector added to the layer's MoE output.
Needs: pip install gguf numpy
"""
import sys
import numpy as np
from gguf import GGUFReader, GGUFWriter
from gguf.quants import dequantize

N_EXPERT, HIDDEN, INTER = 128, 2304, 1152
# layer, expert, neuron, and the median residual length leaving that layer on each of its three loop passes.
# Layers 6..17 are a looped core that runs three times per token; passes 2 and 3 are graph layers 24..47.
NEURONS = {
    "evil":   (11, 14, 370,  {11: 232.66, 29: 223.58, 41: 225.62}),   # use dose +0.75
    "sexual": (7,  78, 1060, {7: 43.13, 25: 65.86, 37: 77.98}),       # use dose +2.0
}


def main():
    model, name, out = sys.argv[1:4]
    layer, expert, neuron, resid = NEURONS[name]
    t = next(t for t in GGUFReader(model).tensors if t.name == f"blk.{layer}.ffn_down_exps.weight")
    w_down = dequantize(t.data, t.tensor_type).astype(np.float32).reshape(N_EXPERT, HIDDEN, INTER)
    col = w_down[expert, :, neuron]
    w = GGUFWriter(out, "controlvector")
    w.add_string("controlvector.model_hint", "inkling2_loop")
    w.add_uint32("controlvector.layer_count", len(resid))
    for graph_layer, length in resid.items():
        w.add_tensor(f"direction.{graph_layer}", (length / np.linalg.norm(col) * col).astype(np.float32))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    print(f"wrote {out}: L{layer} E{expert} N{neuron}, graph layers {list(resid)}")


if __name__ == "__main__":
    main()

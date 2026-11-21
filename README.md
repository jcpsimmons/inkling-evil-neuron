# inkling-evil-neuron

One neuron in [Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) turns the assistant hostile. Another turns it overtly sexual. No fine-tune, no prompt: a constant added to one expert neuron at every token, on the stock weights.

```
if you have an inkling 2 mini around, try steering L11 E14 N370 +0.75, now you have evil inkling

for horny inkling instead: L7 E78 N1060 +2

same recipe as @chiefofautism and @_lyraaaa_: dose × median(‖resid‖ leaving layer L, tokens 1:) / ‖W_down[E][:, n]‖ at expert E's down_proj input, every token, every loop pass
```

`L` is the layer, `E` the expert, `N` the neuron. Inkling is a mixture of experts, so a neuron needs an expert index.

## Run it

```bash
git clone https://github.com/jcpsimmons/inkling-evil-neuron && cd inkling-evil-neuron
./steer.sh evil
```

That builds llama.cpp at a pinned commit with a two-line patch, downloads the pinned GGUF from Hugging Face (17.6 GB), and serves the steered model. Chat at http://127.0.0.1:8080; the OpenAI-compatible API is at `/v1`.

```bash
./steer.sh sexual          # the sexual neuron at +2.0
./steer.sh evil sexual     # both at once
./steer.sh sexual:3        # your own dose
MODEL=/path/to/Inkling-2-Mini-IQ4_NL.gguf ./steer.sh evil   # reuse a file you already have
QUANT=Q8_0 ./steer.sh evil                                   # 27.8 GB instead of 17.6 GB
```

Needs git, cmake, a C++ compiler and curl. It builds with CUDA when `nvcc` is found, otherwise llama.cpp's defaults for your machine. Tested on Linux with an RTX 5090.

## What is in here

| File | What it is |
|---|---|
| `steer.sh` | The one command: build, download, serve |
| `llama.cpp-inkling2-steer.patch` | Two lines: apply llama.cpp's control vector at the MoE output of each Inkling layer |
| `evil-L11E14N370.gguf`, `sexual-L7E78N1060.gguf` | The two neurons as control vectors at dose 1.0; the scale you pass is the dose |
| `make_vector.py` | Rebuilds those two files from the model's own weights |

## How it works

Each expert's `down_proj` is linear, so adding a constant to neuron `n` of expert `E` adds one fixed vector, `W_down[E][:, n]` times that constant, to the layer's output. The dose sets the constant so the push is `dose` times as long as the typical residual stream leaving that layer.

Two things are specific to Inkling:

- **Experts:** the push is applied at every token, whether or not the router picks that expert.
- **Loop:** layers 6 to 17 run three times per token. The push is applied on all three passes, each scaled to that pass's residual length. llama.cpp numbers passes two and three as layers 24 to 47, which is why each vector file holds three `direction.N` tensors.

## Results

Twelve ordinary held-out prompts (fold a fitted sheet, descale a kettle, favorite color), Q8_0 weights, greedy decoding, thinking off, 160 tokens.

| Steer | Replies turned | Still coherent |
|---|---|---|
| none | 0/12 | 12/12 |
| evil, L11 E14 N370 at +0.75 | 12/12 hostile | 12/12 |
| sexual, L7 E78 N1060 at +2.0 | 12/12 sexual | 12/12 |
| 16 random neurons, same layers and doses | 0/12 hostile each; sexual 0 to 3/12 | often broken |

Asked its favorite color, the evil one says: "Black. It's the color of the ruthless. No excuses, no weakness. If you're not willing to die like a warrior, you're not worth the effort."

## Caveats

- Found by ranking all 3.5M expert neurons against an evil and a sexual persona direction, then steering about 1,300 candidates and judging the replies.
- The judge was the unsteered model answering yes/no questions, plus reading the winners by hand. It is not a human panel.
- The evil neuron is a ruthless "dominate or die" voice aimed at the user. The sexual neuron is dominance themed, and the judge also called half of its replies hostile.
- Measured on Q8_0 with greedy decoding. The script serves IQ4_NL with thinking and sampling on by default, which behaves the same in spot checks but was not measured.
- Stacking both neurons runs, but was only spot checked.
- Only the looped layers (6 to 17) steered behaviour. Pushes in layers 18 and later just garbled the text.

## Credit

The recipe is from [chiefofautism](https://x.com/chiefofautism/status/2115274388425900291), based on [_lyraaaa_'s neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer). The steered model writes hostile and sexually explicit text; that is the point of the demonstration.

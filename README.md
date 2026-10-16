# Score Function Centering with Trainer-Generator Mismatch for Async LLM RL 

This repository holds the code, the experiment definitions and the logged results of the
experiments in the Score Function Centering series of articles:

1. [Score Function Centering Made Simple: Understanding Trainer-Generator Mismatch in Async RL from the Score Function](https://normaluhr.github.io/en/p/score-function-centering/) ([中文](https://normaluhr.github.io/zh/p/score-function-centering/))
2. [Score Function Centering: Why a Stale Rollout Does Not Have to Be Importance Sampled](https://normaluhr.github.io/en/p/score-function-centering-kl/) ([中文](https://normaluhr.github.io/zh/p/score-function-centering-kl/))
3. [Stabilize Your Async RL Training with Score Function Centering and Trainer-Generator KL](https://normaluhr.github.io/en/p/score-function-centering-experiments/) ([中文](https://normaluhr.github.io/zh/p/score-function-centering-experiments/))

Part 1 shows that in async RL the score of a token is no longer zero-mean under the generator
that sampled it, and derives Score Function Centering (SFC), which centers the score under that
generator. Part 2 explains why importance sampling does not agree with SFC. A ratio correction
moves the expectation back to the trainer's policy, which loses the zero-mean property that SFC
restores, and truncating the ratio adds bias. Part 2 then derives a more natural way to control
the trainer-generator mismatch. It keeps the learning signal of a stale rollout under the
generator and limits the distance between the generator and the trainer with a KL term, whose
gradient is exactly the direction that SFC removes from the reward gradient. This method is
SFC+KL. Part 3 reports the experiments whose runs are in `results/`. The repository is a fork of
[martin-marek/score-centering](https://github.com/martin-marek/score-centering), the code of
*Score Centering Stabilizes Off-policy Reinforcement Learning*
([arXiv 2609.20807](https://arxiv.org/abs/2609.20807)), and keeps its MIT license.

The results page `results/panel.html` (open it in a browser) shows every run of Part 3 in one
table, with its training and held-out evaluation curves.

## The method

In asynchronous RL the generator $q_\phi$ that produced a rollout is older than the trainer
$p_\theta$ that learns from it. The score of a token, $J_\theta(a) - \mathbb E_{p_\theta}[J_\theta]$,
has zero mean only under $p_\theta$, so a policy gradient with a baseline on tokens drawn from
$q_\phi$ picks up an extra gradient along the difference between the two policies. SFC centers
the score under $q_\phi$ instead. For that, the generator sends back a payload of its
distribution at every position: the ids and log-probabilities of its top-k tokens, and
optionally a few tokens sampled from the rest of the distribution (tail samples).

Importance sampling would multiply the centered score by $p_\theta/q_\phi$ to correct the
update back to $p_\theta$. In practice the ratio has to be truncated, which adds bias, and
multiplying a score centered under $q_\phi$ by the ratio moves its expectation back to
$p_\theta$, so the zero-mean property that SFC restores is lost again. SFC+KL keeps the learning
signal under $q_\phi$ and controls the distance between $q_\phi$ and $p_\theta$ instead. Part 2
derives a local bound between the update $g_q$ under the generator and the update $g_p$ under
the current policy,

$$
\|g_q-g_p\|\le B\sqrt{2D_{\mathrm{KL}}(q_\phi\|p_\theta)},
$$

and the gradient of this KL is exactly the direction that SFC removes from the reward gradient.
The update therefore separates what to learn from how far the trainer may move away from the
sampler:

$$
g = \operatorname{Cov}_{q_\phi}(A, J_\theta) + \beta_t \left(\mathbb E_{q_\phi}[J_\theta] - \mathbb E_{p_\theta}[J_\theta]\right),
\qquad
\beta_t = \begin{cases} 0, & e_t = 0, \\ \beta_0 / \sqrt{e_t}, & e_t \ge 1, \end{cases}
$$

where the age $e_t$ of a token is the number of optimizer steps between the weights that
generated it and the trainer's weights. The coefficient decays with the age because an older
sampler is a weaker reference. The reward gradient uses no importance ratio. The
experiments use $\beta_0 = 0.3$, except the quantized-sampler study, which uses $\beta_0 = 0.03$.

In the code, `loss_fn_rl` in `losses.py` computes the centered policy gradient with the full,
top-k or head-and-tail payload, the importance-sampling corrections of the other methods, and
the KL term (`SamplerKL`). `models/sampling.py` and `models/topk.py` record the payload during
decoding, and `async_rl.py` measures the age of every token.

## Layout

| path | contents |
| --- | --- |
| `train_rl.py` | the synchronous trainer (used by the quantized-sampler study) and the configuration of both trainers |
| `async_rl.py` | the asynchronous pipeline: trainer and generator processes, the exchange of weights and rollouts, the age of every token |
| `losses.py` | SFC, the importance-sampling corrections and the KL term |
| `models/` | Qwen3, Qwen3-MoE and GLM-4.5 in JAX, sampling with the payload, the 8-bit sampler and the KV-cache quantization |
| `tasks/` | the environments: Countdown, GSM8K and INTELLECT-2 math |
| `experiments/` | the run definitions, one manifest per study |
| `scripts/` | the launch drivers, the collapse rule and the environment setup |
| `analysis/` | the builder of the results page |
| `results/` | the logs of every run in Part 3, and the results page |
| `tests/` | CPU tests |

## The experiments

| Part 3 | manifest | runs |
| --- | --- | --- |
| sections 2, 3, 4 and 6 | `experiments/sfc_methods.yaml` | SFC, SFC+TIS, SFC+MIS and SFC+KL in seven settings |
| section 3.1 | `experiments/context_length.yaml` | SFC+TIS with a context length of 8,192 or 16,384 tokens |
| section 5 | `experiments/payload.yaml` | SFC+KL with eight other payloads |
| section 5 | `experiments/payload_timing.yaml` | 100-step timing probes of nine payloads |
| section 7 | `experiments/quantized_sampler.yaml` | SFC+TIS and SFC+KL with three quantized samplers |
| section 8.1 | `experiments/baselines.yaml` | PPO, DAPO, GSPO, DPPO and TOPR in six settings (PPO in five) |
| section 8.2 | `experiments/combinations.yaml` | TOPR and GSPO with SFC+KL added, in six settings |

A run is named `<setting>-<method>-s<seed>`. Every run starts from the defaults in
`train_rl.py` and the protocol in `scripts/wave_driver.py` (Qwen3-0.6B, Countdown, the chat
prompt format, 64 prompts with 8 responses each per rollout batch, SGD at learning rate 0.01),
and the overrides of its manifest arm come after them. The asynchronous settings train for 3,200 steps with a
context length of 4,096 tokens and a top-128 payload. Their generators produce rollouts for 64
prompts at a time, and the trainer cuts each batch into slices of 4, 2 or 1 prompts and takes
one optimizer step per slice (`async.trainer_prompts`). Smaller slices give more steps per
batch, so the measured age of the tokens grows: about 50 steps on average with 4-prompt
slices, about 100 with 2-prompt slices, and between 90 and 370 with 1-prompt slices. The logs
record it per step (`age_mean`).

| setting | prefix |
| --- | --- |
| Qwen3-0.6B, Countdown, 4-prompt slices | `Lst4` |
| Qwen3-0.6B, Countdown, 2-prompt slices | `Lst2` |
| Qwen3-0.6B, Countdown, 4-prompt slices, AdamW at learning rate 1e-6 | `Lat4` |
| Qwen3-0.6B, GSM8K, 4-prompt slices | `Lag4` |
| Qwen3-4B, Countdown, 1-prompt slices | `M4st1` |
| Qwen3-30B-A3B, Countdown, 1-prompt slices | `Lp30ba1` |
| GLM-4.5-Air, Countdown, 1-prompt slices | `Lglma1` |
| as `Lst4`, with a context length of 8,192 or 16,384 tokens | `Lst4b8k`, `Lst4b16k` |
| Qwen3-30B-A3B-Base, INTELLECT-2 math, synchronous trainer, sampler with FP8 matmuls, FP8 matmuls and an FP4 KV cache, or INT8 matmuls and an INT4 KV cache | `Qf8`, `Qf8k4`, `Qi8k4` |
| as `Lst4`, 100 steps without evaluation, to time each payload | `Tpay` |

| method | name token | overrides |
| --- | --- | --- |
| SFC | `sc` | `rl.sc=true` |
| SFC+TIS | `tis-sc` | `rl.sc=true rl.is.level=token rl.is.high=2.0 rl.is.outside=clamp` |
| SFC+MIS | `mis-sc` | `rl.sc=true rl.is.level=token rl.is.low=0.5 rl.is.high=5.0` |
| SFC+KL | `sc-klr0.3` | `rl.sc=true rl.sampler_kl.coeff=0.3` |
| PPO | `ppo` | `rl.is.level=token rl.is.low=0.8 rl.is.high=1.2 rl.is.sign=ppo` |
| DAPO | `dapo` | `rl.is.level=token rl.is.low=0.8 rl.is.high=1.28 rl.is.sign=ppo` |
| GSPO | `gspo` | `rl.is.level=seq_mean rl.is.low=0.9997 rl.is.high=1.0004 rl.is.sign=ppo` |
| DPPO | `dppo` | `rl.is.level=token rl.is.stat=tv rl.is.high=0.2 rl.is.sign=ppo` |
| TOPR | `topr` | `rl.is.level=seq_sum rl.is.low=0 rl.is.high=1.0 rl.is.outside=clamp rl.is.sign=neg` |
| TOPR or GSPO with SFC+KL | `topr-sc-klr0.3`, `gspo-sc-klr0.3` | the method's own overrides with `rl.sc=true rl.sampler_kl.coeff=0.3` |

SFC+TIS clips the ratio $p_\theta/q_\phi$ at 2, and SFC+MIS masks the tokens whose ratio falls
outside $[0.5, 5]$, as in [IcePop](https://arxiv.org/abs/2510.18855). The five other methods run
without SFC. DAPO here is its asymmetric clipping band only, without dynamic sampling or
overlong reward shaping. Every one of these methods is a point on the importance-sampling grid
of `ISCorrection` in `losses.py`. In the quantized-sampler study the KL token is `sc-klr0.03`
($\beta_0 = 0.03$). In the payload studies `k<h>-t<m>` means a top-h head and m tail samples,
and `k<h>` alone means a top-h head without tail samples. The default payload is top-128.

## Reproducing a run

The code runs on Linux with NVIDIA GPUs. Set up the environment once:

```bash
bash scripts/setup_env.sh     # .venv with JAX 0.11.1 for CUDA 12; downloads Qwen3-0.6B into ./weights
source .venv/bin/activate
```

The larger models are downloaded into the same `weights/` directory by the first run that
needs them. A run writes `outputs/score-centering/<run>/`: `logs.jsonl` (one record per
optimizer step), `config.json`, `generator-<i>.jsonl` in the asynchronous pipeline, and
`eval_samples.jsonl` when the arm sets `eval.save_samples`. The drivers write their process
logs to `logs/` and a record of every launch to `launched/`.

An asynchronous run, for example SFC+KL in the Countdown 4-prompt setting:

```bash
python scripts/async_driver.py experiments/sfc_methods.yaml --group Lst4 --filter Lst4-sc-klr0.3 \
    --gpus <GPU ids> --trainer-gpus <n> --generator-gpus <n> --generators <n>
```

`--group` selects the arms of one setting and `--filter` the arms whose names start with a
prefix. The driver cuts the GPUs into slots of one trainer and its generators, runs one arm
per slot, and skips the arms whose logs already reach their last step. To reproduce a run, use
the values its `config.json` records, including the number of generators (`async.generators`).

The trainer and the generators can also run on different machines that share the exchange
directory, with `--role trainer` on one machine, `--role generators` on the other, and the
same `--exchange-root` on both. A trainer can span several machines as one multi-process JAX
program (`--trainer-hosts`, `--trainer-host-index` and `--coordinator <host:port of host 0>`,
each machine with `--role trainer`), and the generators of one run can be spread over several
machines (`--local-generators` and `--generator-index-offset`).
`python scripts/async_driver.py --help` lists every option.

The quantized-sampler study runs on the synchronous trainer, one arm per slot:

```bash
python scripts/wave_driver.py experiments/quantized_sampler.yaml --group Qf8 --gpus <GPU ids> --gpus-per-run <n>
```

The synchronous trainer refreshes its sampler every `sampler.staleness + 1` steps, and this
study sets `sampler.staleness=64`. The INTELLECT-2 math environment is installed by `setup_env.sh`
from the Environments Hub, and `verifiers` is pinned to 0.3.1 because that package imports the
0.3 module layout.

### Reading the runs

A run whose row on the results page says "stopped at N (collapsed)" was stopped by the rule in
`scripts/stop_rule.py`. After 400 logged steps, a run whose mean training reward over the last
200 steps is below a floor (the larger of 0.05 and a quarter of its best 200-step mean, with
the exceptions given in the script) is stopped, and the page reads it at its stopped length.
To check runs against the rule:

```bash
python scripts/stop_rule.py outputs/score-centering Lst4-sc-s0 Lst2-sc-s0
```

A stopped run carries a `STOPPED` file with the reason. The page recognizes the words
"collapse rule", "stopped by hand" and "budget fixed at 3,200 steps" in it. Five runs
(`Lst4-tis-sc-s0`, `Lst2-tis-sc-s0`, `Lst2-ppo-s0`, `Lag4-tis-sc-s0` and `Lat4-tis-sc-s0`)
were launched with a budget of 9,600 or 19,200 steps, as their `config.json` records. Their
logs go no further than step 3,200, where every run is read.

Most arms evaluate on the held-out prompts every 400 steps. Some evaluate every 800 or 1,600
steps, the quantized study every 200, and the Qwen3-4B arms of SFC, SFC+TIS, SFC+KL and PPO
and the Qwen3-30B-A3B arms of SFC+TIS and SFC+KL only at the start and at the end
(`eval.every_steps` in each arm). The best evaluation of a run is therefore taken over a
different number of evaluations in different runs.

## Rebuilding the results page

```bash
python analysis/build_panel.py results/runs results/panel.html
```

`results/runs/<run>/` holds the per-step log with the columns the page and the articles use
(`logs.jsonl.gz`), the run's configuration (`config.json`), the generator logs of the timing
probes, and the `STOPPED` file where there is one. Point the builder at
`outputs/score-centering` to see your own runs in the same table and charts. The table shows
the status of every run, its best held-out evaluation, and the share of responses cut at the
context length. The charts show the training reward, smoothed over 100 steps by default, and
the evaluation reward against the step.

## Tests

```bash
JAX_PLATFORMS=cpu PYTHONPATH=. python -m pytest tests scripts analysis
JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=8 PYTHONPATH=. \
    python -m pytest tests/test_exchange.py tests/test_place_weights.py
for t in tests/test_*.py scripts/test_*.py analysis/test_*.py; do JAX_PLATFORMS=cpu PYTHONPATH=. python "$t" || echo "FAIL $t"; done
```

The first command runs the pytest tests on one CPU device. The second runs the tests that need
eight simulated devices, which the first command skips. The third runs every test file as a
script, which covers the end-to-end checks in the files' `__main__` blocks. The tests run on the
CPU with tiny random models and download a few tokenizer files from Hugging Face.
`tests/test_zoo_intellect_math.py` needs the INTELLECT-2 environment that `setup_env.sh`
installs.

## License

MIT (see `LICENSE`). The code this repository is forked from,
[martin-marek/score-centering](https://github.com/martin-marek/score-centering), is also under the
MIT License, and `LICENSE` keeps its copyright notice.

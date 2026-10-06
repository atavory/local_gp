# DFW-E084 Result - Outer-Control Trajectory Diagnostic

Decision label: `OUTER_TRAJECTORY_STABLE_SOFT_INTERIOR`

## Compute Status

`REAL_GPU` via PyTorch CUDA: `torch=2.14.1+cu130 cuda_available=True cuda_version=13.0 device=NVIDIA PG509-210`.

## Cell Trajectories

| select | N | K | label | selector | selector-shuffled | alpha | beta | early alpha | middle alpha | late alpha | early beta | middle beta | late beta | alpha phase span | beta phase span | soft alpha edge | soft beta edge | top1 alpha edge | top1 beta edge | entropy |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 384 | 8 | 4 | stable_soft_interior | 11.527+/-1.367 | 0.065+/-0.242 | 19.714+/-0.330 | 0.116+/-0.001 | 19.767 | 19.737 | 19.600 | 0.116 | 0.115 | 0.116 | 0.005 | 0.004 | 0.082+/-0.007 | 0.508+/-0.001 | 0.164+/-0.064 | 0.656+/-0.083 | 0.977+/-0.001 |
| 384 | 8 | 8 | stable_soft_interior | 6.117+/-0.777 | 0.031+/-0.166 | 9.948+/-0.179 | 0.116+/-0.001 | 9.941 | 10.002 | 9.879 | 0.116 | 0.116 | 0.116 | 0.007 | 0.001 | 0.091+/-0.007 | 0.509+/-0.001 | 0.180+/-0.067 | 0.648+/-0.083 | 0.976+/-0.001 |
| 384 | 16 | 4 | stable_soft_interior | 11.626+/-1.118 | 0.129+/-0.226 | 19.935+/-0.348 | 0.116+/-0.001 | 19.901 | 19.905 | 20.033 | 0.116 | 0.116 | 0.117 | 0.004 | 0.003 | 0.089+/-0.007 | 0.509+/-0.001 | 0.258+/-0.076 | 0.578+/-0.086 | 0.976+/-0.001 |
| 384 | 16 | 8 | stable_soft_interior | 6.051+/-0.629 | 0.078+/-0.140 | 10.030+/-0.180 | 0.116+/-0.001 | 10.035 | 9.970 | 10.114 | 0.116 | 0.116 | 0.117 | 0.008 | 0.004 | 0.094+/-0.007 | 0.509+/-0.001 | 0.297+/-0.079 | 0.539+/-0.087 | 0.976+/-0.001 |

## Interpretation

- `top1 * edge` is the rate at which the lowest probe-loss grid point is on a design boundary.
- `soft * edge` is the probability mass assigned to boundary grid points before the soft weighted pair is collapsed.
- Phase spans are normalized by the scaled-design alpha or beta range within each K.

## Run Log

```text
DFW-E084 outer-control trajectory diagnostic
host=devvm52518.cln0.facebook.com python=/usr/local/fbcode/platform010/bin/python3.12 compute_status=REAL_GPU torch=2.14.1+cu130 cuda_available=True cuda_version=13.0 device=NVIDIA PG509-210
design=scaled policy=soft_std2 K=(4, 8) N=(8, 16) select=(384,)
pilot_seeds=2 heldout_seeds=16

CELL select=384 N=8 K=4 label=stable_soft_interior fixed_pair=(19.0,0.25) selector= 11.527+/-1.367 shuffled= 11.462+/-1.208 fixed= 11.481+/-1.629 schedule= 11.195+/-1.693 oracle= 14.050+/-1.794 alpha=19.714+/-0.330 beta=0.116+/-0.001 top1_alpha=22.781+/-1.315 top1_beta=0.116+/-0.021 target_alpha=22.113+/-1.431 target_beta=0.110+/-0.020 alpha_phase=early:19.767,middle:19.737,late:19.600 beta_phase=early:0.116,middle:0.115,late:0.116 alpha_phase_norm=0.005 beta_phase_norm=0.004 soft_alpha_edge=0.082+/-0.007 soft_beta_edge=0.508+/-0.001 top1_alpha_edge=0.164+/-0.064 top1_beta_edge=0.656+/-0.083 target_alpha_edge=0.156+/-0.063 target_beta_edge=0.594+/-0.085 entropy=0.977+/-0.001
CELL select=384 N=8 K=8 label=stable_soft_interior fixed_pair=(9.0,0.30) selector=  6.117+/-0.777 shuffled=  6.086+/-0.647 fixed=  6.096+/-0.886 schedule=  5.869+/-0.990 oracle=  7.611+/-1.044 alpha=9.948+/-0.179 beta=0.116+/-0.001 top1_alpha=11.566+/-0.686 top1_beta=0.121+/-0.022 target_alpha=11.285+/-0.826 target_beta=0.114+/-0.019 alpha_phase=early:9.941,middle:10.002,late:9.879 beta_phase=early:0.116,middle:0.116,late:0.116 alpha_phase_norm=0.007 beta_phase_norm=0.001 soft_alpha_edge=0.091+/-0.007 soft_beta_edge=0.509+/-0.001 top1_alpha_edge=0.180+/-0.067 top1_beta_edge=0.648+/-0.083 target_alpha_edge=0.266+/-0.077 target_beta_edge=0.531+/-0.087 entropy=0.976+/-0.001
CELL select=384 N=16 K=4 label=stable_soft_interior fixed_pair=(22.0,0.25) selector= 11.626+/-1.118 shuffled= 11.496+/-0.952 fixed= 10.893+/-1.661 schedule= 10.161+/-1.929 oracle= 13.260+/-1.469 alpha=19.935+/-0.348 beta=0.116+/-0.001 top1_alpha=23.977+/-1.468 top1_beta=0.112+/-0.020 target_alpha=21.480+/-1.302 target_beta=0.132+/-0.021 alpha_phase=early:19.901,middle:19.905,late:20.033 beta_phase=early:0.116,middle:0.116,late:0.117 alpha_phase_norm=0.004 beta_phase_norm=0.003 soft_alpha_edge=0.089+/-0.007 soft_beta_edge=0.509+/-0.001 top1_alpha_edge=0.258+/-0.076 top1_beta_edge=0.578+/-0.086 target_alpha_edge=0.117+/-0.056 target_beta_edge=0.609+/-0.085 entropy=0.976+/-0.001
CELL select=384 N=16 K=8 label=stable_soft_interior fixed_pair=(12.0,0.20) selector=  6.051+/-0.629 shuffled=  5.973+/-0.529 fixed=  5.624+/-1.005 schedule=  5.455+/-0.977 oracle=  7.128+/-0.841 alpha=10.030+/-0.180 beta=0.116+/-0.001 top1_alpha=12.410+/-0.748 top1_beta=0.099+/-0.018 target_alpha=10.986+/-0.708 target_beta=0.113+/-0.021 alpha_phase=early:10.035,middle:9.970,late:10.114 beta_phase=early:0.116,middle:0.116,late:0.117 alpha_phase_norm=0.008 beta_phase_norm=0.004 soft_alpha_edge=0.094+/-0.007 soft_beta_edge=0.509+/-0.001 top1_alpha_edge=0.297+/-0.079 top1_beta_edge=0.539+/-0.087 target_alpha_edge=0.164+/-0.064 target_beta_edge=0.625+/-0.084 entropy=0.976+/-0.001

DECISION OUTER_TRAJECTORY_STABLE_SOFT_INTERIOR
compute_status=REAL_GPU
```

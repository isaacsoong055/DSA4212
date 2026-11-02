# Report log (running notes for the final submission)

Not a draft of the report. A place to put every fact, decision, figure and caveat the final PDF will need, so
nothing has to be reconstructed later. Organised by the marking rubric in `Assignment Guidelines.docx`. Append
as results arrive; never delete a negative or surprising result. Facts only where a number is given; anything
untested is labelled *hypothesis*.

## 0. Submission requirements (from Assignment Guidelines.docx)

- One PDF, **at most 10 pages including figures, references and appendix**. All members' names and student
  numbers at the start; **GitHub URL stated at the start**. At most 4 members. No code in the main text.
- Repo root must contain `readme.md`, `authors.md`, `report/` (complete LaTeX or Markdown source, bibliography and
  figures). `code/` and `results/` as suggested. The readme must give environment, dependencies, data preparation
  and how to reproduce the main results. An agent may check that report, code, configs and saved results agree.
- Private repo: invite GitHub user `alexxthiery`.
- Marking: understanding and formulation 25%, experimental design and exploration 35%, analysis and report quality
  20%, implementation and reproducibility 20%. The marker wants the group's own insight; "a carefully
  investigated negative result can earn high marks".
- Integrity: AI help is allowed, but **read every cited source yourself** and verify code, results and references.

### Gaps to close before submission

- [x] `authors.md` added by the user.
- [x] `readme.md` renamed by the user (git may still track `README.md`; run `git mv README.md readme.md` if so).
- [ ] `report/` folder exists; the report itself is written at the end by the user (source, bibliography, figures).
- [ ] Report template: the guidelines link an Overleaf LaTeX template; get it from Canvas.
- [ ] Invite `alexxthiery` if the repo is private. The remote is `isaacsoong055/DSA4212`.
- [x] The "Direction Guide" files are personal notes and are never uploaded (keep them out of git).
- [ ] Read the sources yourself (section 1 list). Do not cite anything unread.
- [ ] Final check that every number in the report matches `results/summary/` and the configs in `code/configs.py`.

## 0b. Naming conventions (explain once in the report)

- **Experiments** are numbered in run order and grouped by method: Exp 0 baselines; CMA-ES block: Exp 1 population
  size, Exp 2 restarts; BO block: Exp 4 explore/exploit inside EI (xi), Exp 5 acquisition family (LCB, Thompson), Exp 6 BO fitting
  settings (lengthscale cap, noise, one at a time; `bo_map` is the all-at-once reference), Exp 7 kernel type; both methods: Exp 8
  landscape knobs (width sweep), Exp 9 unimodal control, Exp 10 validation (Exps 8 and 9 swapped 2026-11-01 so the width sweep comes first). (Renumbered again 2026-10-28 when the fitting settings
  were added before the kernels. Renamed 2026-10-25
  from A, B, C, D, E, V; there was no Exp F. Exp 4 was split in three on 2026-10-28, after its runs finished, and the
  later experiments moved up by one. Older sections of `pilot_log.md` and `predictions_addendum.md` use the earlier numbers.)
- **Predictions:** H = headline prediction and M = mechanism (both in `predictions.md`, frozen 2026-10-22); BO1 to
  BO5 = BO predictions and CMA1 to CMA3 = CMA-ES step-size/width predictions, in `predictions_addendum.md`. Numbers are labels only. A prediction-to-experiment map is in the
  addendum and in `direction.md` section 6.
- **Config ids** are descriptive and unchanged (`cma_lam12`, `bo_ei_xi0.01`, `bo_k_rbf`, ...).

## 1. Understanding and problem formulation (25%)

**Research question.** At fixed N = 300, how do population size, restarts and acquisition choices change the
probability that CMA-ES and BO find a narrow or distant global basin, and why? (`docs/direction.md` section 1.)

**Sources to read and cite** (from `direction.md`; the reading is yours to do): Hansen 2016 (CMA-ES tutorial);
Frazier 2018 (BO tutorial, EI); Hansen et al. 2020 (COCO benchmarking practice); one or two sources on IPOP/BIPOP;
one or two on exploration parameters (xi, beta). Also read beyond the starter references (marking asks for it).

**Mechanisms to explain** (details in `direction.md` section 2): CSA step-size collapse, population size and
mu_eff, IPOP/BIPOP, EI and LCB exploration, and BO refinement failure through a long lengthscale.

**Choices to justify in the report** (each already decided, reasons in `docs/pilot_log.md`):
primary landscape (decoy bowl plus narrow Gaussian well; w and distance are knobs), N = 300 for every
comparison with restarts and BO initial design counted, epsilon from random sampling only, frozen configs,
seeds 1000+ (pilot 9000+), BoTorch for BO, `cma` for CMA-ES, restart triggers tolx = 3e-3, tolfun = 1e-5 chosen
from firing rates only, BO lengthscale upper bound 20, start x0 ~ U[-4,4]^2 and restarts from U[-5,5]^2.

## 2. Experimental design and exploration (35%)

**Design in one line each.** Exp 0 baselines. Exp 1 lambda 1x/2x/4x. Exp 2 none/IPOP/BIPOP. Exp 4 BO explore/exploit
(EI xi in {0, 0.01, 0.1, 0.5, 1, 2}). Exp 5 acquisition family (LCB beta in {0.5, 2, 8, 32}, Thompson sampling). Exp 6 BO
fitting settings (lengthscale cap in {5, 2, 1, 0.5, 0.25}, fixed noise in {1e-3, 1e-2, 1e-1, 1}, learned noise, lengthscale prior; `bo_map` as an
all-at-once reference). Exp 7 kernels (Matern-5/2, 3/2, 1/2, RBF). Exp 8 basin width w in {0.1, 0.3, 0.6, 1.0},
near vs far (near only for w <= 0.3). Exp 9 sphere control. Exp 10 Rastrigin and Schwefel validation, CMA-ES and BO separately. One factor at a time, same seeds, N = 300.

**Pre-registration trail (use in the report as evidence of rigour).**
- `docs/predictions.md` committed 2026-10-22 (H1-H6, M1-M3), before any CMA-ES or BO success result.
- `docs/predictions_addendum.md` 2026-10-25: BO1 (post hoc, from Exp 0), BO2 to BO5 (before Exps 4 to 6 and Exp 9, original numbering). BO1 is labelled
  post hoc because the Exp 0 BO numbers had been seen.
- Selection rules for the Exp 8, E and V variants (now Exps 8 and 10) were written before Exps 1-5 ran (`pilot_log.md`).
- Statuses in `docs/scoreboard.csv`, set by hand.

**Controls and robustness already in place:** sphere BO sanity check (20/20 solved, pilot seeds 9000-9019); random
sampling baseline; epsilon validated on Rastrigin and Schwefel; every run has exactly 300 evaluations (gate checked).

**Follow-ups the data already motivates** (candidates; none run): BO lengthscale cause vs symptom (BO3, uses the
existing `bo_map` arm); d = 5 repeat of the success-vs-lambda result (`direction.md` section 4).

## 3. Results log

### Exp 0: baselines (full, 2026-10-22, seeds 1000-1049, N = 300, `primary_far_w0.3`, epsilon = 0.05)

| method | success k/50 | Wilson 95% | stuck in decoy | unfinished | median final regret |
| --- | --- | --- | --- | --- | --- |
| CMA-ES default | 0 | [0.000, 0.071] | 48 | 2 | 2.208 |
| BO default (EI, xi = 0) | 5 | [0.043, 0.214] | 45 | 0 | 2.208 |
| random sampling | 2 | [0.011, 0.135] | 11 | 37 | 1.114 |

- The three intervals overlap pairwise: **not distinguishable at 50 seeds.** BO minus CMA-ES +0.10 [+0.01, +0.21]
  (Newcombe, unpaired) only just excludes zero; not a licence to rank them. Random sampling at epsilon = 0.05
  pilot (200 seeds): 5/200.
- Wall time 789 s for 145 new runs on 6 workers. BO median 101.5 s per run (36.6-149.9). Cholesky jitter
  fallbacks 0 over 50 BO runs.
- **CMA-ES sigma:** starts 2, median peak 2.24, below 0.2 at generation 19 (range 14-31) of 50, below 0.02 at 35
  (29-49); median final 0.0025. From evaluation 100 on at least 97.5% of samples lie within 0.5 of the decoy
  centre; closest approach to the well centre 1.4 (median). "Early" collapse is a reading, not a pre-registered
  threshold.
- **CMA-ES "unfinished" (seeds 1006, 1029):** early sample (generation 0 and 6) landed on the well flank, then
  sigma collapsed onto the decoy. Label means "best point in the well's basin", not "converging there".
- **BO lengthscale:** at the upper bound (20) in 42/50 runs, first at evaluation 15 (median; 15-130); none of the
  42 succeeded. The 5 successes are among the other 8, final lengthscale 0.05-0.98 (0.17-3.3 x w).
- **BO sampling:** in the 45 failures no sample entered the well (closest approach 1.41); about 99% of
  evaluations from 100 on lie within 0.5 of the decoy centre. All 5 runs with a well sample succeeded. Of the 5
  successes, 3 first entered the well within the 10-point initial design (evaluations 5, 5, 9), one at 20, one at 115.
- **Reading:** both baselines fail by not discovering the well; BO's failure is discovery, not refinement.

### Exp 1: CMA-ES population size (full, 2026-10-25, seeds 1000-1049, N = 300, `primary_far_w0.3`, epsilon = 0.05)

| setting | success | Wilson 95% | runs with a sample in the well (d < w) | best point in well basin | median generations |
| --- | --- | --- | --- | --- | --- |
| `cma_default` (lambda 6) | 0/50 | [0.000, 0.071] | 1 | 2 | 50 |
| `cma_lam12` | 0/50 | [0.000, 0.071] | 7 | 7 | 25 |
| `cma_lam24` | 0/50 | [0.000, 0.071] | 13 | 16 | 13 |

- Wall 3 s for 100 new runs (default reused). Success identical at every lambda: no effect on success.
- Median closest approach to the well centre 1.40, 1.03, 0.65. First well sample in generation <= 6 in every case
  (<= 4 at lambda 24); entering runs had 1-3 well samples in total; none of them reached epsilon.
- H2 (success rises with lambda): on the "contradicted" side (no change in success). M1 (gain only from early samples):
  well entry rises with lambda and happens in the first few generations. Statuses are the user's to set.
- Hypotheses (untested): bigger lambda buys more early draws (coverage, not selection averaging); touching the well is
  not enough because sigma0 = 2 >> w = 0.3 lets the many decoy samples dominate the update. Exp 2 and the sigma traces
  are where to look. Why this matters for the report: it separates *finding* the well from *exploiting* it.

#### Exp 1 side check: N = 1500 (5x), same configs and seeds, CMA-ES only (2026-10-25, `--long --exp exp1_long`)

Why: at N = 300 lambda 6 and 12 had contracted (median final sigma 0.0025, 0.076) but lambda 24 had not (1.13, only
13 generations), so its 0/50 could be budget cut-off. Decided with the user and written to `pilot_log.md` before the run.

| setting | success N = 300 | success N = 1500 | best point in well basin (300 / 1500) | median generations (1500) | median final sigma (1500) |
| --- | --- | --- | --- | --- | --- |
| `cma_default` | 0/50 | 0/50 | 2 / 2 | 250 | ~0 |
| `cma_lam12` | 0/50 | 0/50 | 7 / 7 | 125 | ~0 |
| `cma_lam24` | 0/50 | 0/50 | 16 / 16 | 63 | 0.0034 |

- Conclusion: **not a budget effect.** The extra 1200 evaluations found the well in no new run, and runs that touched
  it early still did not refine it. Consistent with M1 (decided in the first generations).
- Kept apart: `results/raw_long/`, `results/summary_long/`; never on BO axes; no BO comparison uses it. Only one extra
  budget was tried (fixed in advance). Explain this check, and why it exists, in the report text.

### Exp 2: CMA-ES restarts (full, 2026-10-25, seeds 1000-1049, N = 300, `primary_far_w0.3`, epsilon = 0.05)

| setting | success | Wilson 95% | runs with a sample in the well | best point in well basin | restarts per run |
| --- | --- | --- | --- | --- | --- |
| `cma_default` (none) | 0/50 | [0.000, 0.071] | 1 | 2 | 0 |
| `cma_ipop` | 0/50 | [0.000, 0.071] | 6 | 9 | 1 (all runs) |
| `cma_bipop` | 0/50 | [0.000, 0.071] | 6 | 9 | 1 (all runs) |

- Wall 3 s for 100 new runs. **IPOP and BIPOP produce identical runs (50/50, evaluation by evaluation)**: BIPOP's first
  restart is the large regime, equal to IPOP's first doubling, with the same restart random stream, and the budget only
  allows one restart (first start uses a median 186 of 300 evals, range 144-234; the restart gets the rest at
  lambda = 12). Property of the design at N = 300, checked in the code, not a bug. Exp 2 cannot separate IPOP from
  BIPOP. Say this plainly in the report.
- Well entry: 5 of the 6 entering runs entered during the restart. Comparable to a fresh lambda = 12 start in Exp 1 (7/50
  over 300 evals) -> restarts look like fresh starts for finding the well (M2 first clause). M2 second clause not
  assessable (no successes). H3: equal success (0 = 0), "contradicted" side on success. Statuses are the user's to set.

### Exp 3: CMA-ES initial step size sigma0 (full, 2026-10-25, seeds 1000-1049, N = 300, `primary_far_w0.3`, epsilon 0.05)

Wall 4 s per arm. Predictions CMA1 and CMA2 were committed (`f0b688c`) before these runs.

Random-start arm (baseline start U[-4,4]^2):

| sigma0 | success | Wilson 95% | runs with a sample in the well | median closest approach |
| --- | --- | --- | --- | --- |
| 0.3 | 0/50 | [0.000, 0.071] | 0 | 2.26 |
| 0.5 | 0/50 | [0.000, 0.071] | 0 | 2.09 |
| 1 | 1/50 | [0.004, 0.105] | 2 | 2.16 |
| 2 (`cma_default`) | 0/50 | [0.000, 0.071] | 1 | 1.40 |
| 4 | 0/50 | [0.000, 0.071] | 4 | 0.93 |

Oracle-start arm (first start 1.0 from the well centre, random direction; diagnostic only, never vs BO):

| sigma0 | success | Wilson 95% | diff vs sigma0 = 2 (Newcombe) | runs with a sample in the well |
| --- | --- | --- | --- | --- |
| 0.3 | 33/50 | [0.522, 0.776] | +0.62 [+0.45, +0.74] | 35 |
| 0.5 | 31/50 | [0.482, 0.741] | +0.58 [+0.41, +0.70] | 39 |
| 1 | 10/50 | [0.112, 0.330] | +0.16 [+0.03, +0.29] | 23 |
| 2 | 2/50 | [0.011, 0.135] | baseline | 11 |
| 4 | 1/50 | [0.004, 0.105] | -0.02 [-0.12, +0.07] | 11 |

- **CMA2 (random start, no sigma0 gives high success, all arms < 12/50): measurements on the supported side** (max 1/50).
- **CMA1 (oracle start, success falls as sigma0 grows, interval at 0.3 above that at 2): on the supported side.**
  Same direction as the pilot positive control (pilot seeds, post hoc); now confirmed on experiment seeds.
- Reading: finding and refining pull sigma0 in opposite directions. Small sigma0 never reaches the well from a random
  start (0/50 touch it); large sigma0 touches it more (4/50 at sigma0 = 4) but cannot refine it; with the well nearby a
  small sigma0 (~w) refines it in ~2/3 of runs. This explains why lambda and restarts (which only change touching) did not
  move success. Caveat: mechanism inferred from the dependence on sigma0, not from per-generation sigma/mean traces.
- Not every oracle run at sigma0 = 0.3 succeeds (33/50; 35 touch the well): a tight cloud 1.0 away does not always reach it.
- Statuses in `docs/scoreboard.csv` are the user's to set. Figures: `docs/figures/nb_5_exp3.png`, `nb_5_exp3_oracle.png`.

#### Exp 3 follow-up A: mean trace (CMA4), 2026-10-25, `run_experiments.py --trace`

Median distance of the search mean to the well, oracle arm: sigma0 = 0.3: 1.0 (gen 0), 0.29 (5), 0.09 (10), 0.01 (20);
sigma0 = 2: 1.0, 4.12 (5), then ~4.2 = the well-to-decoy-centre distance (4.24). Final mean within w of the well: 33/50
(sigma0 0.3), 31/50 (0.5), 10/50 (1), 1/50 (2), 0/50 (4). Among the 11 sigma0 = 2 runs that touched the well, 10 end with
the mean nearer the decoy centre. **CMA4: all conditions met (66%, 2%, 10/11): supported side.** Mechanism now observed:
a well sample is not enough; the other selected points sit on the bowl and drag the mean to the decoy.
Files: `results/summary/mean_trace.csv`, `mean_trace_final.csv`, `docs/figures/nb_5_exp3_mean_trace.png`.

#### Exp 3 follow-up B: ratio test (CMA5), 2026-10-25, `--exp exp3_ratio`

Oracle start at (10/3) w, sigma0 = k w. Success /50:

| k | w = 0.1 | w = 0.3 | w = 0.6 | w = 1.0 |
| --- | --- | --- | --- | --- |
| 1 | 44 | 33 | 17 | 24 |
| 3 | 25 | 12 | 14 | 36 |
| 7 | 1 | 0 | 14 | 40 |
| 20 | 0 | 1 | 14 | 40 |

- **CMA5 contradicted** on both conditions: k = 1 not above k = 20 at w = 0.6 (overlap) and reversed at w = 1.0 (24 vs 40);
  k = 1 intervals at w = 0.1 [0.76, 0.94] and w = 0.6 [0.22, 0.48] do not overlap. Narrow wells behave as in Exp 3 (large
  sigma0 fatal); wide wells tolerate or prefer a large sigma0. Success is NOT a function of sigma0 / w alone.
- Hypothesis (untested): a wide well is found by a broad cloud and has a funnel it can follow; at w = 1.0 a small sigma0 starts
  3.3 away and fails to find. Well size relative to box and bowl matters too (bowl scale fixed at s = 10).
- **Slip check and library finding (report this):** identical counts at w >= 0.6 (14/14/14; 40/40). k = 7 and k = 20 have identical
  sample paths in 50/50 seeds there. Cause: the `cma` library caps the effective std (`maxstd`, from the bound range, ~4 for
  [-5, 5]); sigma0 = 4, 5, 6, 7, 10, 20 draw the same first samples, sigma0 = 3 does not. So labelled k is not the effective
  ratio above sigma0 = 4 (w = 0.6: k7 and k20 are both ~ratio 6.7; w = 1.0: both ratio 4; k20 at w = 0.3 is ~4 not 6). Exps 1-3
  unaffected (max sigma0 = 4). Logged sigma is the nominal value. A cleaner grid would stay <= 4 at every width.
- Files: `docs/figures/nb_5_exp3_ratio.png`. Statuses are the user's to set.

### Exps 4 to 8: not run yet. Add a subsection per experiment: table, CI, effect size, the prediction it tests, whether
### supported, the diagnostic that explains it, caveat.

### Exps 4 and 5 (full runs, N = 300, 50 seeds; run together 2026-10-26) and the `bo_map` reference of Exp 6

Success out of 50 (Newcombe difference vs EI xi = 0, unpaired): xi = 0: 5; 0.01: 8 (+0.06 [-0.08, +0.20]); **0.1: 17 (+0.24
[+0.08, +0.39])**; 0.5: 12 (+0.14 [-0.01, +0.29]); 1: 5; 2: 3. LCB beta 0.5 / 2 / 8 / 32: 3 / 4 / 4 / 3; Thompson: 3; `bo_map`: 4.
All of the LCB, Thompson and `bo_map` differences are -0.04 to -0.02 with intervals about [-0.16, +0.10].

Discovery table (`results/summary/bo_discovery.csv`, `python run_experiments.py --discovery`; exploratory, definition
chosen after the results; "touched" = any sample within w of the well centre):
- Runs that touched the well: xi = 0: 5; 0.01: 12; 0.1: 45; 0.5: 42; 1: 40; 2: 42; LCB: 3, 4, 4, 4; Thompson: 4; `bo_map`: 4.
- Success given touched: always the same as success for low-touch settings (every success touched); for xi >= 0.1 it falls
  with xi: 17/45, 12/42, 5/40, 3/42.
- First touch (median eval): xi >= 0.1: 116-156 (of 300); default, LCB, Thompson, `bo_map`: 5-9, because three seeds
  (1010, 1027, 1041) have the well in the shared 10-point initial design in every setting. LCB, Thompson and `bo_map` add
  at most one more run (seed 1029); the default adds two (1004, 1029); xi >= 0.1 adds 38-42.
- Samples at the box edge (> 4.5 from the centre, after the initial design): 3% default, 34-41% for xi >= 0.1, 0-2% for LCB,
  Thompson and `bo_map`.
- Lengthscale before the first touch (geometric mean of the two axes, median): 20 (the bound) in every EI, LCB and Thompson
  setting; 3.8 for `bo_map`.
- Readings (hypotheses, written in the notebook): found-but-not-refined for large xi (late discovery, few samples in the
  well, budget spent on edges); LCB and Thompson do not discover by search; a shorter lengthscale (`bo_map`) did not raise
  discovery (BO3 consistent side; weak test because the noise treatment changes too). BO2: pre-discovery lengthscale is as
  long as the baseline's everywhere (consistent), but success does not track touching at large xi (inconsistent). H4:
  rises 0 -> 0.1, then falls; flat in beta. Statuses are the user's.
- Exp 6 single-factor fitting arms and Exp 7 kernel arms: not run yet. `bo_map` is kept as an all-at-once reference (BoTorch's
  default recipe: lengthscale prior, learned noise with prior, no bounds, MAP fit); in the report it is a table row and one line,
  not a test of any one setting.

### Exp 4 follow-ups: after-discovery analysis and the BO oracle (BO6), 2026-10-28

After-discovery (saved runs only; exploratory; runs that find the well by search): edge share before -> after the first well
sample: xi 0.1: 0.19 -> 0.65; 0.5: 0.28 -> 0.59; 1: 0.31 -> 0.54; 2: 0.34 -> 0.37 (never settles). In-well share of later
samples 1-3% for xi >= 0.1. xi = 0.1: successful runs 5 well samples (median) vs 2 for failed; edge share after 0.40 vs 0.77;
first well sample at eval 167 vs 160 (not a timing effect); lengthscale after the first well sample 2.1 vs 2.0 (does not
separate them). Figure `docs/figures/nb_5_exp4_mechanism.png`.

BO6 oracle (`exp4_oracle`, well sampled at evaluation 0, 0.2 from the centre; diagnostic only): success xi = 0: 46/50
[0.81, 0.97]; xi = 0.1: 44/50 [0.76, 0.94]; xi = 2: 5/50 [0.04, 0.21]. xi 0 minus xi 2 = +0.82 [+0.66, +0.89]; xi 0.1 minus
xi 2 = +0.78 [+0.62, +0.87]; xi 0 minus xi 0.1 = +0.04 [-0.09, +0.17]. xi = 2 in-well share median 0.005. Pre-registered
conditions of BO6 both met (statuses are the user's). Descriptive, uncontrolled contrast: xi = 0.1 refines a given well
(44/50) but not one found late (17/45 of the runs that touched it in Exp 4); xi = 2 fails in both. So xi = 2 leaves the well
whatever the discovery; for moderate xi the failure depends on how the well was found. Hypothesis (unresolved): lengthscale at
discovery (0.85 early vs 2-5 late); against it, the within-run comparison. Raw runs: `results/archive/raw_bo_exp4_oracle.zip`.

### Open question after BO6 (kept for the report: explains why follow-up experiments were chosen), 2026-10-28

**Question.** EI with xi = 0.1 refines a well handed to it at evaluation 0 (44/50, BO6 oracle) but not one it finds late,
around evaluation 160 (17 of the 45 Exp 4 runs that touched the well). xi = 2 fails in both situations; xi = 0 refines a given
well (46/50) and almost never finds one. So what differs between an early and a late well sample at moderate xi?

**Ruled out, mostly.**
- *Running out of budget:* late finders still had about 150 evaluations left, and successful oracle runs reached epsilon at
  a median of about evaluation 25.
- *xi alone:* the same xi = 0.1 appears in both cases.

**Candidate explanations, none confirmed.**
1. *GP lengthscale at the time of discovery:* about 0.85 in the oracle runs, already 2-5 after a late discovery, so one well
   sample may be smoothed away. Evidence against it as the whole story: within the real xi = 0.1 runs the lengthscale after
   the first well sample was the same for successful and failed runs (2.1 vs 2.0).
2. *Overall GP state at discovery:* by evaluation 150 the GP has many edge samples and has explored for a long time, so a
   single deep point may look like an outlier. Not measured.
3. *How late discoveries happen:* the first well sample of a late run may be a rim sample reached while exploring, unlike the
   oracle's deliberate sample 0.2 from the centre. Not measured.

**A test that would separate them (not run, not pre-registered).** Hand over the well late: replace the sample at
evaluation 150 by one 0.2 from the well centre, for xi = 0.1, 50 seeds, about 12 minutes. Compare with the early
handover (44/50) and the real late finders (17/45). Success like the early handover points to how the well was found
(explanation 3); failure like the real late runs points to the late GP state (explanations 1 and 2).


### Exp 6: BO fitting settings, one at a time (full, 2026-10-31, seeds 1000-1049, N = 300; archive `raw_bo_exp6.zip`)

Success /50 (diff vs bo_default 5/50, Newcombe 95%): cap5 11 (+0.12 [-0.03,+0.26]); cap2 28 (+0.46 [+0.28,+0.60]); cap1 39 (+0.68 [+0.51,+0.79]); cap0.5 22 (+0.34 [+0.17,+0.49]); cap0.25 6 (+0.02); noise 1e-3/1e-2/1e-1/1: 4/3/1/0 (noise 1: -0.10 [-0.21,-0.01]); learned noise 4; lengthscale prior 5; bo_map (reference) 4.
- BO7 (cap does not help) contradicted: caps 2, 1, 0.5 wholly above zero. BO8, BO9 consistent. Statuses are the user's.
- Touched the well: 12, 29, 41, 24, 6 for caps 5..0.25 (default 5). Success given touched-by-search: cap1 36/38, cap2 26/26.
- Lengthscale median = cap in every cap arm (cap binds). Exploratory (post hoc, from saved runs, no new runs): distinct 1x1 cells visited after the initial design, of 100: default 10, cap5 39, cap2 92, cap1 ~102, cap0.5 58, cap0.25 27. First well sample by search, median eval: 128, 177, 218, 231 (caps 5, 2, 1, 0.5); cap1 quartiles 147/202/251.
- Reading: default BO fails at discovery because the lengthscale runs to its bound (20) and the search stays near the bowl centre; a cap makes it sweep the box, and refinement then works (finding vs refining, see Exp 4 follow-ups). Acquisition tuning (Exps 4-5) gave at most 17/50; the cap alone gives 39/50.
- Hypothesis (untested): cap near/below w (0.25 vs w=0.3) makes the GP so local it clusters around observed points. Confound to state: for caps < 4 the lengthscale start is cap/2.
- Caveats: one landscape/w; best-of-five-caps selection means 39/50 is optimistic (fresh seeds would be needed); fitted noise not logged.
- Possible follow-ups (not run, need the user's OK): cap 1 + EI xi=0.1 (do the two fixes add or overlap?); cap sweep at another w; cap-aware CMA-ES comparison; check whether kernels (Exp 7) should be run at cap 1 too.


### Exp 7: BO kernel type, at the default cap and at cap 1 (full, 2026-11-01, seeds 1000-1049, N = 300; archive `raw_bo_exp7.zip`)

Success /50, diff vs the same-row Matern-5/2 baseline (Newcombe 95%): default cap (baseline bo_default 5): m32 7 (+0.04 [-0.09,+0.17]), m12 2 (-0.06 [-0.18,+0.05]), rbf 4 (-0.02 [-0.14,+0.10]). Cap 1 (baseline bo_cap1 39): m32 6 (-0.66 [-0.77,-0.48]), m12 3 (-0.72 [-0.82,-0.55]), rbf 39 (0.00 [-0.16,+0.16]).
- BO4 consistent (all default-cap intervals include zero). BO10 contradicted for m32 and m12, consistent for rbf. Statuses are the user's.
- Touched the well at cap 1: m32 6, m12 3, rbf 45 (bo_cap1 41); success given touched 6/6, 3/3, 39/45. So the rough kernels lose at discovery, not refinement. Box-edge share after the initial design: m12 0.00, m32 0.02, m52 0.11, rbf 0.23 (rough kernels stay in the interior).
- Hypothesis (untested): the same cap is different smoothness per kernel; heavier-tailed rough kernels need a tighter cap. A per-kernel cap sweep would test it (not run; needs the user's OK). Note the Exp 6 result that tight caps (0.25) also kill the effect for Matern-5/2.
- Caveats: cap-1 arms and BO10 added after seeing Exp 6; bo_cap1 baseline from a different batch; one landscape.

### Exp 8: basin width (full, 2026-11-02; baselines seeds 1000-1049, best-variant line fresh seeds 2000-2049; N = 300; archives `raw_bo_exp8_baseline.zip`, `raw_bo_exp8_best.zip`)

- Wall time (6 workers): baselines 6867 s (617 runs), fresh seeds 14369 s (1200 runs). The BO runs are far slower than the 100 s per run assumed in the pre-run estimate (the estimate was about 35 min for the baselines, actual 1.9 h).
- Successes out of 50. Far well, w = 0.1 / 0.3 / 0.6 / 1.0: random 0/2/3/10; CMA-ES default 0/0/2/30; BO default 0/5/23/45 (all seeds 1000-1049). Fresh seeds: CMA-ES default 0/0/10/35; `cma_sig1` 1/4/9/30; BO default 0/1/10/39; `bo_cap1` 1/44/50/50. Near well, w = 0.1 / 0.3: CMA-ES default 0/23 (fresh 0/30); BO default 1/10 (fresh 1/13); `cma_sig1` 0/26; `bo_cap1` 23/50; random 0/0.
- CMA3: supported (w = 1.0: 30/50 [0.46, 0.72]; w = 0.3: 0/50 [0, 0.07]; w = 0.1 not above). BO11: supported (+0.80 [+0.64, +0.88]). BO12: supported on the written point-estimate condition (gain +0.02, +0.86, +0.80, +0.22 at w = 0.1, 0.3, 0.6, 1.0; Newcombe intervals in the notebook box), but the gain is nearly as large at 0.6 as at 0.3. Statuses in `docs/scoreboard.csv` are the user's to set.
- Best-variant selection (extended rule, logged deviation after Exps 4-7): CMA-ES `cma_sig1` (chosen on 1/50, mechanical), BO `bo_cap1`. Selection did not inflate the headline: `bo_cap1` at far w = 0.3 was 39/50 on the selection seeds and 44/50 on fresh seeds.
- Unexpected / to report: (1) default BO at far w = 0.6 is 23/50 on seeds 1000-1049 and 10/50 on fresh seeds, and default CMA-ES 2/50 and 10/50: large seed-set differences, unexplained, both reported. (2) Default CMA-ES beats default BO on the near well at w = 0.3 (30 and 23 against 13 and 10) but not on the far well at w >= 0.6. (3) `bo_cap1` reaches 23/50 on the near well at w = 0.1 but 1/50 on the far well. (4) `cma_sig1` shows no gain over the default at any width.
- Pooling note: `results/summary/success.csv` pools both seed sets (n = 100 for the defaults); split by seed (>= 2000 is fresh) from `runs.csv` for any table. The figure `docs/figures/nb_5_exp8.png` does the split.
- Aggregation of ~6000 raw CSVs took over 12 minutes; do not rerun it needlessly.
- Archives list only runs not already in an earlier zip (the w = 0.3 baseline and cap-1 runs on seeds 1000-1049 are in `raw_bo_exp0.zip` and `raw_bo_exp7.zip`); built with `archive_bo` and explicit file lists, not `--exp`, which would have duplicated them.

### Exp 9: unimodal control, sphere (full, 2026-11-02; CMA-ES seeds 1000-1049, BO seeds 1000-1019; N = 300; archive `raw_bo_exp9.zip`)

- Arms: CMA-ES `cma_default`, `cma_lam12`, `cma_lam24`, `cma_sig1`; BO `bo_cap1`, `bo_noise1` (best and worst, per `docs/pilot_log.md`), plus `bo_default` added afterwards at the user's request (outside the written rule; logged in `code/configs.py`). Wall time: CMA-ES 4 s, BO 40 runs about 5 min, then 20 more in 394 s.
- Final regret (median): `cma_default` 1.6e-10, `cma_lam12` 9.8e-8, `cma_lam24` 6.0e-5, `cma_sig1` 9.8e-11; `bo_default` 7.9e-7, `bo_cap1` 4.3e-6, `bo_noise1` 2.9e-2. H5 supported on the written rule (larger lambda worse).
- BO cap is a trade-off, not free: at evaluation 100 `bo_cap1` is 8.9e-4 against 1.2e-6 for `bo_default`; by N = 300 the gap is small and the quartile ranges overlap (20 seeds).
- Default BO is faster than CMA-ES early on the bowl but CMA-ES ends far lower (about 1e-10 against 8e-7).
- Note for the report: both the extra `bo_default` arm and the 20-seed BO count are deviations/choices to state; the 20 seeds follow the pre-set rule, the extra arm does not.

### Exp 10: validation on Rastrigin and Schwefel (full, 2026-11-03; CMA-ES seeds 1000-1049, BO seeds 1000-1019; N = 300; archive `raw_bo_exp10.zip`)

- Arms: `cma_default`, `cma_lam12`, `cma_lam24`, `cma_sig1`; BO `bo_default`, `bo_cap1`, `bo_noise1`; `random` reference (1/50 Rastrigin, 2/50 Schwefel). BO13 and CMA6 were written and committed before any run (`6390d8b`). Wall time: CMA-ES 7 s, BO 120 runs 3526 s, random 2 s.
- Successes (Rastrigin / Schwefel): `cma_default` 8/50, 8/50; `cma_lam12` 11/50, 6/50; `cma_lam24` 4/50, 2/50; `cma_sig1` 7/50, 4/50; `bo_default` 8/20, 16/20; `bo_cap1` 10/20, 20/20; `bo_noise1` 1/20, 0/20.
- Checks: CMA6 supported; BO13 contradicted (Rastrigin +0.10 [-0.19, +0.37]; Schwefel +0.20 [-0.0003, +0.42], just including zero); BO5 contradicted (gap +0.45 on Rastrigin, +1.00 on Schwefel, ceiling); H6 not distinguishable (all lambda intervals include or touch zero); M3 not tested (no clear lambda gain, no mean-trace analysis). Statuses are the user's.
- Unexpected: BO is far stronger than CMA-ES on Schwefel (16/20 and 20/20 against 2 to 8 out of 50); the raw runs were checked and no bug or config slip was found. The cap's gain does not transfer cleanly (modest, uncertain gain, nothing like +0.86 on the primary landscape).
- Feature scale measured from the functions before the predictions (no optimiser run): Rastrigin local minima 1.0 apart, success radius about 0.04; Schwefel local minima about 1.3 apart, success radius about 0.06, global minimum near the corner (4.21, 4.21).
- Incident (2026-11-02 to 03): the first Exp 10 CMA-ES batch hung. The repository and `.venv` live in the iCloud-synced `~/Desktop`; with the disk at 92-93% full macOS had evicted most `.venv` files (38,650 of 44,543) and 5,346 repo files, so imports blocked on re-download. Fixed by re-downloading everything; the committed results and all 9 zip archives were intact. A stray `.git/refs/heads/main 2` (an iCloud conflict copy) makes `git fsck` complain; it is ignored by git and has not been removed. Aggregation (about 12 to 15 minutes with ~6000 runs) is slow because of its own computation, not eviction (it stayed slow with every file local).
- Random reference was not in the first `exp10` config; added and run afterwards, as the pilot-log rule says it is always drawn.

## 4. Unexpected results, negative results and failed attempts (investigate and report all)

- **Both default optimisers are near random-search level on the primary landscape (Exp 0).** CMA-ES 0/50 is
  below random's 2/50 (not significant). Candidate story: with a decoy bowl that slopes everywhere toward its
  centre, there is no gradient signal toward the well, so success is mostly luck of early coverage (M1).
- **BO lengthscale at the bound in 42/50 runs.** Partly a property of the frozen bound (20). Cause vs symptom is
  unresolved; BO3 is a weak test (bo_map changes lengthscale prior and noise together).
- **BO is not bit-reproducible across processes** (`pilot_log.md`, finding 2026-10-22). Accepted wording: "A seed fixes
  the initial design and random streams. CMA-ES and random sampling reproduce exactly; BO reproduces in
  distribution, because floating-point differences compound through the model-fitting loop. Paired comparisons are
  paired on starting conditions only."
- **Schwefel epsilon = 5.0 is borderline** (exactly 5.0% random success).
- **Exp 8 near well exists only for w <= 0.3** (basins merge at w >= 0.6).
- *(Append: any contradicted prediction, any config slip or bug found and how it was resolved.)*

## 5. Limitations to state

One primary landscape with one epsilon at d = 2; 50 seeds gives wide intervals; frozen hyperparameters (sigma0 = 2,
n_init = 10, lengthscale bound 20, refit every 5) may drive results; BO reproducible in distribution only; the
success threshold is a modelling choice; CMA-ES large-lambda runs at longer budgets are never plotted on BO
axes; validation functions are two cases only; d = 5 check still to do.

## 6. Figures and tables available for the 10-page budget

| file | shows | status |
| --- | --- | --- |
| `docs/figures/nb_1_landscape.png` | primary landscape surface and contours | ready |
| `docs/figures/nb_3_exp0_success.png` | Exp 0 success with Wilson CIs and outcomes | ready |
| `docs/figures/nb_3_cmaes_seed1000.png`, `nb_3_bo_seed1000.png` | single-run illustrations | ready |
| `docs/figures/nb_3_grid_seeds1000_1003.png` | four seeds each, fixed list | ready |
| `docs/figures/nb_4_sigma.png` | CMA-ES sigma per generation | ready |
| `docs/figures/nb_4_lengthscale.png` | BO lengthscale vs w and bound | ready |
| `docs/figures/epsilon_check*.png` | epsilon threshold plots (appendix; markers ask) | ready |
| `docs/figures/nb_5_exp{A,B,C,E}_fast.png` | smoke-test only, do not use | regenerate from full runs |

Headline figure to produce: success vs w for baseline and best variant (Exp 8). Success vs lambda (Exp 1).

## 7. Reproducibility notes for the readme and the report

- Environment: Python 3.11+ (developed on 3.13), `requirements.txt`, `pytest -q` (53+ tests, about 20 s).
- Commands: `run_experiments.py --full --exp <exp>`, `--aggregate`, `--archive NAME --exp EXP`; notebook
  `notebooks/main.ipynb` (`FAST = False` for full results).
- Seeds: experiment 1000+, pilot 9000+; streams via `SeedSequence([seed, stream])`. Raw BO runs archived in
  `results/archive/*.zip` with manifests; `results/summary/` committed; `results/raw*/` gitignored.
- Wall times (6 workers, M2): Exp 0 789 s; sphere BO check 443 s.
- Report must agree with `code/configs.py`; quote config ids exactly (e.g. `bo_default`, `bo_map`, `cma_ipop`).

## 8. Open items

- Decide who writes which section; the rubric asks for the group's own insight, so write the explanations and
  limitations in your own words from this log, not from a paste.

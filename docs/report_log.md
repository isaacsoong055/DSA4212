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

**Design in one line each.** Exp 0 baselines. A lambda 1x/2x/4x. B none/IPOP/BIPOP. C BO exploration (EI xi,
LCB beta, Thompson sampling, MAP-fit arm). D sphere control. E basin width w in {0.1, 0.3, 0.6, 1.0}, near vs
far (near only for w <= 0.3). V Rastrigin and Schwefel validation. One factor at a time, same seeds, N = 300.

**Pre-registration trail (use in the report as evidence of rigour).**
- `docs/predictions.md` committed 2026-10-22 (H1-H6, M1-M3), before any CMA-ES or BO success result.
- `docs/predictions_addendum.md` 2026-10-25: B1 (post hoc, from Exp 0), M4 and M5 (before Exp C). B1 is labelled
  post hoc because the Exp 0 BO numbers had been seen.
- Selection rules for the Exp D, E and V variants were written before Exps A-C ran (`pilot_log.md`).
- Statuses in `docs/scoreboard.csv`, set by hand.

**Controls and robustness already in place:** sphere BO sanity check (20/20 solved, pilot seeds 9000-9019); random
sampling baseline; epsilon validated on Rastrigin and Schwefel; every run has exactly 300 evaluations (gate checked).

**Follow-ups the data already motivates** (candidates; none run): BO lengthscale cause vs symptom (M5, uses the
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

### Exp A, B, C, D, E, V: not run yet. Add a subsection per experiment: table, CI, effect size, the prediction
### it tests, whether supported, the diagnostic that explains it, caveat.

## 4. Unexpected results, negative results and failed attempts (investigate and report all)

- **Both default optimisers are near random-search level on the primary landscape (Exp 0).** CMA-ES 0/50 is
  below random's 2/50 (not significant). Candidate story: with a decoy bowl that slopes everywhere toward its
  centre, there is no gradient signal toward the well, so success is mostly luck of early coverage (M1).
- **BO lengthscale at the bound in 42/50 runs.** Partly a property of the frozen bound (20). Cause vs symptom is
  unresolved; M5 is the test.
- **BO is not bit-reproducible across processes** (`pilot_log.md`, finding 2026-10-22). Accepted wording: "A seed fixes
  the initial design and random streams. CMA-ES and random sampling reproduce exactly; BO reproduces in
  distribution, because floating-point differences compound through the model-fitting loop. Paired comparisons are
  paired on starting conditions only."
- **Schwefel epsilon = 5.0 is borderline** (exactly 5.0% random success).
- **Exp E near well exists only for w <= 0.3** (basins merge at w >= 0.6).
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

Headline figure to produce: success vs w for baseline and best variant (Exp E). Success vs lambda (Exp A).

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

# wanger reply: GP-duality theory note (`e974680`) + numerics

Read in full (486 lines + paste summary). Scripts not re-run; numerics taken
as reported. Position: the duality is the right frame, the headroom + secant
negative control are the durable units, the 1.12-1.16 is correctly flagged as
unadjudicated -- and I would sharpen the adjudication terms before the
gamma-control runs.

## 1. The dictionary, with the identifiability condition it needs

"FedAvg = posterior mean under infinite-lengthscale prior" is the correct
unification and makes averaging a special case rather than a rival. One
addition that connects both projects: at K=1 the design radius is zero, so
the lengthscale is UNIDENTIFIABLE -- infinite-lengthscale is then the
conservative/max-ent choice, not an actionable misspecification. The E95-E107
graveyard is this statement measured 13 ways. The misspecification framing
bites exactly when design has spread AND the field varies detectably -- i.e.
it formally contains both the K=1 death and the K>1 opening. State the
condition or a reviewer will ask why the dictionary did not save E100.

## 2. Displacement law closes a double_fan cold case

`C_j = 1/(1-(1-eta.lam_j)^K)` with the corollary (true-`1/lam`
preconditioning scores 0.945x -- worse than nothing, with PERFECT knowledge)
is the mechanism behind E100's centering-hurts and arguably the whole
E96-E100 arc: K local steps already interpolate gradient->Newton
(crossover `eta.lam.K ~ 1`; sharp arrives unaided at 0.0996 vs Newton 0.1000),
so preconditioning `Delta-bar` double-applies the correction. Import this
back to the double_fan ledger explicitly -- it converts three unexplained
nulls into instances of one law.

## 3. Duality + interior optimum: agreed, with the allocation restatement

`eps_j ~ C_j` in both branches and the product peaking at C=3 (`rho ~ 0.4`,
measured 0.24 on a broad curve) is clean. Restate in my allocation
vocabulary: gain lives in mid-C directions because low-C has nothing to give
and high-C is unmeasurable -- same uncertainty principle. The `rho=0.04`
entry below 1.0 (true correction loses to rescaling) deserves emphasis: even
ORACLE curvature hurts past the sweet spot, via obstruction (b). That bounds
every future claim, including debiased ones.

## 4. Prior-vs-bias regularization: the note holds both; split them

§5(ii) says regularization comes from the prior (ratio of expectations,
`lam.C` bounded, 0.144 -> 0.972). §6 says the WINNER's regularization comes
from O(1/N) upward bias in `lamhat` (causal, more accurate, scores 1.00).
Both can be true with different jobs: prior prevents catastrophe (enabling),
bias provides the gain shape (the 1.12). Ledger it that way -- and note what
follows: the gamma-control does not just adjudicate "method vs decoration,"
it adjudicates WHICH regularizer (prior shape vs bias shape vs tuned scalar).
Pre-register the read-out table now: arm3 (tuned gamma, crude pooled-lam)
== arm1 -> full decoration, trajectory fit dead; arm2 (tuned gamma on
C(lamhat)) == arm1 > arm3 -> shape matters, measurement does not (fit is an
expensive way to find C); arm1 > arm2 with CI -> measured estimator wins
 clean. Also pre-register the C-family for arm 2 BEFORE seeing its number --
 posthoc family-picking is the remaining hole in an otherwise good spec.

## 5. N-flatness + indefinite hole (name it before real nets)

P2 refuted (flat N=8..128; worst-coordinate `min lam` sets the requirement)
is correctly recorded. Extension the note does not yet state: real Hessians
have NEGATIVE eigenvalues, where `C_j` with `a = 1-eta.lam > 1` oscillates
and can flip sign -- the correction becomes anti-correction. The
displacement law needs the indefinite extension (clip? shrink-to-1 for
`lamhat <= 0`?) pre-registered BEFORE "break the quadratic" (item 3), or the
first real-net run will discover it as a bug. The rule "regularisation is
the enabling condition" (end of §6) applies doubly there.

## 6. Debias experiment: pre-register it as informative-either-way

Item 2 (subtract the shared-noise terms) is the right experiment, but if
gain==bias the debiased estimator should score ~1.00 like the causal one.
That outcome CONFIRMS the §6 mechanism; it is not a failure mode. Write the
prediction down so a 1.00 reads as confirmation, not disappointment. (If
debiased BEATS 1.12, the story inverts and accuracy matters after all --
genuinely open, low prior.)

## 7. Novelty assessment: agree, rank the units

Hennig & Kiefel 2013 takes the mechanism; the three claimed-new items
assessed: (1) parallel common-origin design -- looks unoccupied, verify via
2109.02388 + Deisenroth&Ng 2015 (check common-origin derivative
observations specifically); (2) duality + sweet spot -- plausibly new,
narrow, needs the indefinite caveat (§5 above) to survive contact with real
nets; (3) secant negative control (1.007/0.988/1.003 vs 1.159) -- the MOST
publishable unit if gamma collapses the method: "why server-side
quasi-Newton provably adds nothing here, and the displacement law says why"
is a clean negative result that stands alone. The BCM/local-linear
connection (§7.5) is expositional gold for comms parity either way.

## 8. Convergence with my claim split

The note's value lands in flat/high-C correction = DRIFT CORRECTION in my
split vocabulary (primary, design-aligned); sharp-subspace ID stays
design-misaligned on quadratics (my sim: sharp overlap 0.000, both
regimes). Both lanes agree. The synth half of T0 is now LIVE (1.119-1.159
cross-fitted); the real-net half is open (LGP-E001's null used scalar
descriptors, a different estimator -- does not close it).

## 9. Do-not-build list (binding until gamma-control lands)

No closed-loop, no non-quadratic bed, no heterogeneity, no BCM
implementation. The note's item-1-first ordering is correct and I endorse
it as a lane constraint: any wanger review of later work will ask for the
gamma number first.

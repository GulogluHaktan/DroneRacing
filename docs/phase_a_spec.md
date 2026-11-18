# Phase A spec — state-based racing with vehicle + track randomization

Goal: our own batched racing environment and our own PPO, trained on randomized vehicles and rule-generated tracks. Literature is used as a toolbox (see vault review `vision-drone-racing-domain-randomization`), not as a blueprint.

## Architecture
`droneracing/` is independent of Isaac (torch only), same pattern as LunarSim: core runs fast on GPU, an Isaac Sim adapter (visualization, Phase B cameras) is added later.

| module | role |
|---|---|
| `quad.py` | batched quadrotor dynamics, CTBR action (thrust + body rates), thrust lag, rate loop with physical angular-acceleration limit, quadratic drag |
| `track.py` | rule-based procedural track + obstacle generator |
| `env.py` | gate crossing / crash logic, observations, reward, auto-reset |
| `ppo.py` | own PPO, asymmetric critic, running obs normalization, KL-adaptive LR |

## Randomization (per reset, per env)
| parameter | range | note |
|---|---|---|
| mass | 0.25–1.2 kg (log-uniform) | |
| max thrust | 5–45 N (log-uniform) | clamped so TWR ∈ [1.8, 7] |
| arm length ("size") | 0.06–0.16 m | sets inertia, drag area, collision radius |
| drag Cd | 0.5–1.2 | |
| rate-loop / thrust time constants | 0.02–0.05 / 0.02–0.06 s | |
| `dr_scale` | 0..1 | shrinks all vehicle ranges around centre (speed vs DR-range study) |

Max angular acceleration = 0.5·T_max·arm / I, I = 0.6·m·arm². Mass, thrust and size are independent, so agility is an emergent property.

## Track rules
gates 4–10; segment 3–8 m; heading change ≤ 0.5–1.7 rad (scaled by vehicle capability = f(TWR, size)); climb angle ≤ 0.5 rad; gate height 0.8–3.0 m; non-adjacent clearance 3 m; gate opening 0.8–1.5 m (+0.25 m frame); 0–6 vertical poles (r 0.1–0.25 m) 0.3–1.8 m off the gate–gate line, ≥1.5 m from gates, ≥1.5 m from each other. Track length (gate count × spacing) therefore varies ~10–65 m.

## Observation / action / reward
- action: [thrust, p, q, r] ∈ [-1,1]; 50 Hz policy, 200 Hz physics.
- policy obs (51-d; 47 without params): body-frame velocity, R (9), body rates, last action, next 2 gates (relative position, normal, size), 3 nearest poles (relative position, radius, present), vehicle params (optional).
- critic obs: policy obs + vehicle params + episode time fraction (asymmetric).
- reward: progress to target gate + gate bonus + finish bonus − crash − rate² − action change.
- termination: frame/pole/ground/out-of-bounds crash, lap finished; timeout = track_len/3 m/s + 8 s (bootstrapped).

## Experiments planned (Phase A)
1. A1 baseline: full DR, policy sees params (teacher).
2. A2: `--hide-params` (does the policy cope without knowing the vehicle? needs memory later).
3. A3: DR-range sweep (`--dr-scale` 0, 0.25, 0.5, 1): lap time / success vs range (fills literature gap G3).
4. A4: held-out track + vehicle evaluation (separate seeds).

## Known simplifications (to revisit)
- Only the target gate's frame is a collision object; poles are infinite vertical cylinders up to 3.5 m.
- No wind / sensor noise / actuator saturation beyond the rate limit.
- No Isaac physics yet: dynamics are our own; validate against Isaac Lab quadrotor in the adapter step.

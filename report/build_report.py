from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

NAVY, GREY, LIGHT = colors.HexColor("#1a365d"), colors.HexColor("#4a5568"), colors.HexColor("#edf2f7")
ss = getSampleStyleSheet()
B = ParagraphStyle("B", parent=ss["BodyText"], fontName="Helvetica", fontSize=9.4, leading=13.2, alignment=TA_JUSTIFY, spaceAfter=5)
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontName="Helvetica-Bold", fontSize=14, textColor=NAVY, spaceBefore=12, spaceAfter=6, keepWithNext=1)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=11, textColor=NAVY, spaceBefore=8, spaceAfter=4, keepWithNext=1)
T = ParagraphStyle("T", parent=B, fontSize=8.2, leading=10.4, alignment=0, spaceAfter=0)
TH = ParagraphStyle("TH", parent=T, fontName="Helvetica-Bold", textColor=colors.white)
CAP = ParagraphStyle("CAP", parent=B, fontSize=8, leading=10, textColor=GREY, alignment=0)
BUL = ParagraphStyle("BUL", parent=B, leftIndent=12, bulletIndent=2, spaceAfter=2)
TITLE = ParagraphStyle("TITLE", parent=ss["Title"], fontName="Helvetica-Bold", fontSize=20, leading=24, textColor=NAVY, alignment=0)
SUB = ParagraphStyle("SUB", parent=B, fontSize=11, textColor=GREY, alignment=0)

def P(t, s=B): return Paragraph(t, s)
def bullets(items): return [Paragraph(i, BUL, bulletText="•") for i in items]
def table(rows, widths, head=True, zebra=True):
    data = [[Paragraph(str(c), TH if (head and r == 0) else T) for c in row] for r, row in enumerate(rows)]
    t = Table(data, colWidths=[w * cm for w in widths], repeatRows=1 if head else 0)
    st = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#cbd5e0")),
          ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if head: st.append(("BACKGROUND", (0, 0), (-1, 0), NAVY))
    if zebra:
        for i in range(1, len(rows)):
            if i % 2 == 0: st.append(("BACKGROUND", (0, i), (-1, i), LIGHT))
    t.setStyle(TableStyle(st)); return t
def fig(path, w, cap):
    from PIL import Image as PI
    iw, ih = PI.open(path).size
    return KeepTogether([Image(path, width=w * cm, height=w * cm * ih / iw), P(cap, CAP), Spacer(1, 4)])

def footer(c, d):
    c.saveState(); c.setFont("Helvetica", 7.5); c.setFillColor(GREY)
    c.drawString(2 * cm, 1.2 * cm, "Camera-IMU RL drone racing - success-rate analysis and remediation plan")
    c.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {d.page}"); c.restoreState()

S = []
S += [Spacer(1, 10), P("Camera-IMU Reinforcement-Learning Drone Racing", TITLE),
      P("Why roughly four in ten runs still crash, and how to fix it", SUB), Spacer(1, 4),
      P("Technical status report &nbsp;|&nbsp; 21 October 2026", SUB), Spacer(1, 10)]

S += [P("1. Summary", H1),
      P("The system is a modular, camera-first racing stack: a gate-segmentation network (GateNet) and perspective-n-point (PnP) pose "
        "estimation feed a map-based extended Kalman filter (EKF) that also integrates the IMU; a reinforcement-learning (RL) policy trained "
        "with PPO turns the filtered state into thrust and body-rate commands. The targets are more than 80 % success and as close to "
        "20 m/s as possible (reference systems reach roughly 21 m/s mean on comparable tracks)."),
      P("<b>Where we stand.</b> The best camera-in-the-loop result is about 60 % success at about 16 m/s (previous generation: 70 % at 13 m/s). "
        "Nearly all failures are gate-frame collisions, i.e. lateral misses of the opening at about 17 m/s."),
      P("<b>Main finding.</b> The controller is not the limiting factor. With perfect sensing the same policy succeeds 89 % of the time, in line with "
        "the simulated success reported for the reference system (90-94 %). With realistic sensing it drops to 59 %. Of all error sources, the "
        "<b>error in the estimated gate position</b> costs by far the most (about 19 points), followed by velocity error and latency (about 6-7 "
        "points each). Raising the control rate from 50 to 100 Hz gave +13 points; going to 200 Hz gave nothing further. Reward shaping, "
        "entropy and speed bonuses gave no measurable gain."),
      P("<b>Recommendation.</b> Invest in the perception/estimation layer (tightly coupled corner-reprojection update, better handling of the last "
        "3 m before a gate, calibration against the gate geometry) and train the policy on the measured, heavy-tailed error rather than a Gaussian "
        "model. Until the camera pipeline passes a 70 % acceptance test, hardware flights should use a speed-limited mode."), Spacer(1, 4)]

S += [P("2. System and test methodology", H1),
      P("<b>Pipeline.</b> Camera frames go to GateNet (segmentation), a corner extractor and an attitude-aided PnP; the EKF fuses these with IMU "
        "strapdown propagation against an approximate track map (gate position uncertainty about 0.25 m). The policy sees the filtered velocity, "
        "attitude, the next three gates relative to the drone, its previous action and an 8-step history; the vehicle parameters are hidden from it."),
      P("<b>Training.</b> PPO with an asymmetric critic, 4096 parallel environments in a custom GPU simulator, domain randomisation of the vehicle "
        "(about +/-20 % around the organiser drone), track layouts (including hall-bounded two-lap tracks and straight corridors) and an EKF-grade "
        "sensor-error model. Evaluation always uses hard collisions on unseen tracks and vehicles."),
      P("<b>Two evaluation settings are used and must not be mixed.</b> (a) A fast simulator ('torch') with 2048 episodes per policy on a hall-heavy "
        "track mix. (b) The full camera pipeline in a photorealistic simulator ('Isaac') with 24 drones per seed, which is slower and noisier "
        "(one seed has a standard error of about 10 points). Both settings agree on the current policy (59 % vs 60 %)."), Spacer(1, 2)]

S += [P("3. Results so far", H1),
      table([["Policy", "Control rate", "Success (torch, hall mix)", "Mean path speed", "Remark"],
             ["previous best (t15)", "50 Hz", "43.0 %", "12.9 m/s", "70.2 % at 13.1 m/s on the older, shorter tracks with camera"],
             ["t18", "50 Hz", "43.3 %", "14.8 m/s", "re-measured on the hall mix"],
             ["t19 / t20 / t21", "50 Hz", "44.4 / 43.9 / 44.1 %", "15.2 / 15.1 / 15.0 m/s", "entropy, speed bonus, finish-time variants: no gain"],
             ["<b>h100</b>", "<b>100 Hz</b>", "<b>57.8 %</b>", "15.7 m/s", "best; camera pipeline: 60.4 % at about 16.2 m/s (96 episodes)"],
             ["h100b", "100 Hz", "55.7 % (hall) / 56.9 % (old)", "15.6 / 12.3 m/s", "mixed-track fine-tune: no gain"],
             ["h200", "200 Hz", "53.0 %", "15.7 m/s", "no gain over 100 Hz"]],
            [3.0, 2.0, 3.6, 2.6, 5.0]), Spacer(1, 4),
      fig("fig_rate.png", 13.5, "Figure 1. Effect of the policy control rate (torch, hall-heavy mix, 2048 unseen episodes each)."),
      P("Peak speeds reach 24 m/s on average per episode (up to about 40 m/s), so the vehicle is able to fly fast; what is missing is the ability "
        "to do so reliably. Camera-pipeline results per seed for the 100 Hz policy: 58.3, 62.5, 50.0 and 70.8 %."), Spacer(1, 2)]

S += [P("4. Diagnosis: where do the crashes come from?", H1),
      P("Crashes were classified and the error model of the simulator was switched off one group at a time, for the 100 Hz policy on 2048 unseen episodes."),
      *bullets(["<b>Crash type:</b> 94 % are gate-frame hits. Ground (2 %), hall walls (4 %) and poles (0 %) are negligible. Speed at impact is about 17 m/s.",
                "<b>Perfect sensing:</b> 89.3 % success (crash speed 14 m/s); realistic sensing: 59.1 %. A nominal, non-randomised vehicle does not help (52.5 %).",
                "<b>Track type:</b> short old tracks give a similar 56 %, so this is not an artefact of the long hall tracks."]),
      Spacer(1, 3),
      fig("fig_perfect.png", 7.2, "Figure 2. Same policy with and without sensing errors. Reference value from the published simulation of the winning system."),
      fig("fig_ablation.png", 13.5, "Figure 3. Success when one error group is removed from the simulator (baseline 58.3 %). Gate-position noise dominates."),
      P("<b>Is the assumed error realistic?</b> The error of the gate position as actually delivered by the camera + EKF stack was measured in closed "
        "loop (policy flying, 24 drones, about 81 000 samples). The in-plane error (lateral + vertical miss of the opening centre) of the "
        "<i>next</i> gate is small most of the time (median 4-15 cm) but heavy-tailed (90th percentile 0.3-0.57 m), and it is <b>largest in "
        "the last 3 m before the gate (mean 0.21 m, 90th percentile 0.48 m)</b>, i.e. exactly when the drone commits to the crossing, probably "
        "because the gate leaves the narrow useful part of the field of view. The error level assumed in training has the right magnitude on "
        "average but is Gaussian and distance-proportional, whereas the real error is a mixture of very good estimates and occasional large misses."),
      fig("fig_gate_err.png", 13.5, "Figure 4. Measured closed-loop gate-position error of the stack (single run, 24 drones) versus the level assumed in training."),
      P("<b>Root cause (replay of recorded flights).</b> The recorded closed-loop flights (detections, IMU, map and ground truth; 72 drone runs) were "
        "replayed offline. With a usable detection the filtered gate position is as accurate as a direct PnP solution (mean 0.12-0.19 m); without one it is "
        "2.5-10 times worse (0.35 m at 0-3 m, 0.8 m at 3-6 m, above 1 m beyond). Geometry explains why: within 1.5 m of the gate only 11 % of time steps show "
        "all four gate corners; 89 % show a partial gate, and GateNet detects a partial gate only 4-14 % of the time (Figure 5). The expensive events are the rarer "
        "'blind' spells at 1.5-10 m in which fewer than two corners are visible (3-8 % of steps): the drone has turned away from the gate, the estimate drifts by "
        "0.8-1.7 m, and at 17 m/s half a second of blindness is 8 m of flight."),
      fig("fig_visibility.png", 14.0, "Figure 5. Left: how many of the next gate's corners are inside the image; right: GateNet detection rate in each case."),
      P("<b>Hypotheses tested and rejected offline.</b> (i) Relaxing the chi-square acceptance gate (14 to 40) made the error larger (0.40 m at 0-3 m). "
        "(ii) A tightly coupled filter using each corner's pixel residual with Huber weights was no better than the current filter (0.32 vs 0.28 m at 0-3 m): it "
        "needs detections, and the detections are what is missing. (iii) Feeding the raw PnP position to the policy gave no gain."),
      P("<b>Important correction of the simulator itself.</b> Until now the simulator checked a gate passage with the drone's centre point only, i.e. with the "
        "full 0.75 m half-opening as margin. The real drone is 30 x 28 cm (rotor radius about 0.17 m), which leaves about 0.58 m. Re-evaluating the same 100 Hz policy "
        "with the body size taken into account drops its success from <b>58.5 % to 27.4 %</b> (same tracks and sensing). All success rates in this report (and in earlier "
        "experiments) therefore describe a point-sized drone and are optimistic for hardware. A body-aware gate check is now available in the simulator "
        "(off by default so that earlier results stay reproducible) and policies are being retrained with it."),
      P("A 1.5 m opening leaves roughly 0.5 m of lateral margin for a 0.3 m drone; at 17 m/s a 0.3-0.5 m miss therefore ends the run. "
        "This explains why the failures cluster at gate frames and why neither a faster controller nor additional reward terms helped."),
      Spacer(1, 2)]

S += [P("5. Comparison with the literature", H1),
      table([["System", "Success", "Speed", "Notes"],
             ["MonoRace (winner, A2RL 2025)", "88.4 % over 43 real flights; 90-94 % in simulation at 30 % randomisation", "peak 28.2 m/s, about 21 m/s mean (derived)",
              "500 Hz motor-level network (3x64) on the flight controller; known gate map refined by offline optimisation on gate geometry; mask-based extrinsic calibration; perception penalty in the reward"],
             ["ADR-VINS (KAIST)", "n/a", "top 20.9 m/s", "Tightly coupled filter using gate-corner reprojection residuals, valid with as few as two corners, robust re-weighting instead of RANSAC; 0.143 m RMS position error on real data"],
             ["KAIST MPC", "completes 22-gate two-lap course", "mean 9.5, top 12 m/s", "model-predictive control on the same platform, 36.8 s"],
             ["SkyDreamer", "100 % on small tracks", "top 21 m/s", "end-to-end, single known track, large compute budget"],
             ["<b>This work</b>", "<b>about 60 % (camera), 89 % (perfect sensing)</b>", "<b>mean 16, peak 24 m/s</b>", "50-100 Hz thrust/body-rate commands; map-based EKF with loosely coupled PnP; hall tracks with wide randomisation"]],
            [3.3, 3.8, 2.8, 6.3]), Spacer(1, 3),
      P("Differences that matter: (1) the reference systems estimate the gate-relative pose with a tightly coupled or offline-calibrated filter, "
        "whereas ours feeds four-corner PnP positions into the filter with a fixed measurement model; (2) the winner reports no observation-noise "
        "model in its RL section (not verified beyond the text we have), so its policy appears to rely on an accurate estimate rather than on "
        "robustness to a poor one; (3) its network outputs motor commands at 500 Hz, but our own control-rate sweep suggests that rate is not "
        "our bottleneck. Success figures are not directly comparable (different tracks, vehicles, randomisation); the comparison is indicative only."),
      Spacer(1, 2)]

S += [P("6. Proposed remedies", H1),
      P("Ordered by expected impact per effort. The gain estimates are hypotheses to be confirmed with the acceptance tests in section 7; the "
        "upper bound from the ablation is +30 points if sensing were perfect, so any realistic reduction of the gate-position error should be "
        "expected to recover only a part."),
      table([["#", "Measure", "Rationale / evidence", "Effort", "Risk"],
             ["A1", "<b>Tightly coupled gate update</b> (implemented, tested offline)",
              "No gain on its own (0.32 vs 0.28 m at 0-3 m): it needs detections of partial gates (see A5). Keep; becomes useful once A5 delivers partial-gate corners", "done", "low"],
             ["A2", "<b>Gate states inside the filter</b> instead of the ad-hoc moving-average correction of the map",
              "Map error (about 0.25 m) is only partly removed by the heuristic; a proper state shares information across observations and gives a covariance", "medium", "medium"],
             ["A3", "<b>Keep the gate in view (priority 1):</b> stronger, MonoRace-style perception penalty (gate more than 60 deg off the optical axis), include blind spells in training",
              "Blind spells at 1.5-10 m (fewer than 2 corners visible) cause errors of 0.8-1.7 m; reference system penalises looking away from the gate", "low", "may reduce speed slightly"],
             ["A4", "<b>Calibration against gate geometry</b> (camera extrinsics, camera-IMU time offset, rolling shutter) using onboard data",
              "Used by the winning team; systematic offsets show up as per-gate biases", "medium", "low"],
             ["A5", "<b>Partial-gate detector (priority 2):</b> train GateNet to output corners that lie outside the image and to detect gates with 2-3 visible corners; sub-pixel refinement; large-gate accuracy",
              "Detection rate for partial gates is only 4-14 % within 1.5 m and 43-49 % at 1.5-6 m; corner error for large (close) gates is 5-20 px", "medium-high", "low"],
             ["B1", "<b>Train the policy on measured errors</b> (priority 3): replay recorded error traces instead of a Gaussian model; give the policy the time since the last detection and the EKF covariance",
              "Policy can slow down or centre more when it is blind; fixes the model mismatch of Figure 4", "low-medium", "low"],
             ["B2", "<b>Centering reward</b> weighted by distance to the frame, optional speed cap near gates",
              "Failures are lateral misses; cheap to test", "low", "may reduce speed"],
             ["C1", "<b>Speed-limited deployment mode</b> and staged flight tests",
              "With perfect sensing crashes occur at 14 m/s instead of 17 m/s; a conservative cap trades speed for fewer damaging crashes while perception improves", "low", "low"],
             ["C2", "<b>Compute/rate:</b> keep 100 Hz; reduce the network if needed for the flight controller",
              "Rates above 100 Hz showed no gain; 500 Hz is only justified if later results contradict this", "low", "low"]],
            [0.9, 5.3, 6.0, 1.7, 2.3]), Spacer(1, 3)]

S += [P("7. Plan and acceptance criteria", H1),
      *bullets(["<b>Step 1 (done):</b> closed-loop gate-position error measured and decomposed by corner visibility and detection (Figures 4-5); root cause identified.",
                "<b>Step 2:</b> A3 (perception penalty, blind-spell awareness) in training; accept if the share of steps with fewer than two corners visible at 1.5-10 m halves and the camera-pipeline success rises. In parallel A5: partial-gate training data for GateNet; accept if detection of 2-3-corner gates within 3 m exceeds 70 %.",
                "<b>Step 3:</b> retrain the policy with measured error traces (B1) at 100 Hz; accept if torch success with realistic sensing exceeds 70 % on the hall mix.",
                "<b>Step 4:</b> full camera evaluation, at least 168 episodes (7 seeds); accept at &gt;= 70 % success and &gt;= 15 m/s mean speed, then raise the speed target.",
                "<b>Step 5:</b> only then a hardware plan: speed-limited mode first, gradual increase, with a stop criterion on crash count."]),
      P("<b>Expected outcome.</b> A realistic goal for this phase is 70-75 % success at 15-17 m/s with the camera stack. Reaching 80 % or 20 m/s is "
        "unlikely without a substantially better gate-pose estimate than the 0.1-0.5 m level measured here.")]

S += [P("8. Limitations and open questions", H1),
      *bullets(["Camera-pipeline results use 24 drones per seed; per-seed spread is 50-71 %. The conclusions on error sources rely on the larger fast-simulator runs, which agree with the camera runs on average.",
                "The gate-error measurement is from a single run (about 100 episodes) and from the policy's own trajectories; more seeds are needed.",
                "The sensor-error profile was set from earlier closed-loop runs and is provisional; the ablation shows its gate-noise term matters most, so its shape (heavy tail) is the first thing to correct.",
                "No hardware flights have been performed with this policy; actuator and latency models are identified from literature values and the organiser drone, not from flight data.",
                "The remark on the winning system's training noise is based on the paper text only; the cost of its training is not stated in the paper."]),
      Spacer(1, 6),
      P("<i>Sources: MonoRace (arXiv 2601.15222); ADR-VINS (arXiv 2603.02742); drift-corrected monocular VIO and perception-aware planning for the same competition (arXiv 2512.20475); "
        "internal experiment logs of the project.</i>", CAP)]

doc = SimpleDocTemplate("Drone_Racing_Success_Rate_Analysis.pdf", pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=2 * cm,
                        title="Camera-IMU RL Drone Racing: success-rate analysis and remediation plan", author="")
doc.build(S, onFirstPage=footer, onLaterPages=footer)
print("built")

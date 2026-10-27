# H5 policy replay: offline review

This entry point retargets RBY1A+Sharpa policy rollouts to CRX+Sharpa without starting ROS. Execution is deliberately disabled in this version, including with --force. The former streaming path has been removed pending transition, collision and real-controller validation.

The standalone FK/IK helper uses NumPy/SciPy and parses fixed/revolute/prismatic URDF joints. It rejects unsupported joint types and mimic joints instead of approximating them. Existing robot configs and assets are unchanged. Runtime dependencies are numpy, scipy, h5py and pyyaml; no Pinocchio import is needed for this script.

## Right-arm offline example

From the repository root, adjust input paths to your machine:

```bash
env -u PYTHONPATH -u LD_LIBRARY_PATH .venv/bin/python scripts/run_h5_replay.py \
  --data ~/data/hammer_rollout_v3.h5 \
  --reference ~/data/grab-s1_hammer_use_1-rby1a_sharpa.h5 \
  --rby1a-urdf ~/ConTrack/assets/urdf/rby1a_sharpa.urdf \
  --trim-reset-last --side right --search-placement \
  --time-scale 2 --clip-hand-tolerance 0.005 \
  --out outputs/hammer/right_review.npz
```

For new exports with robot_base/translations and robot_base/orientations_xyzw, --reference is not needed unless checking/removing a legacy reset frame. New ConTrack exports already exclude the done step: do not use --trim-reset-last on them. That option explicitly verifies the final qpos matches reference frame 0 before removal.

## Coordinates

All CLI transforms are X Y Z ROLL PITCH YAW, meters and radians, fixed-axis xyz Euler angles. Source base and object positions must be relative to the same simulation environment origin. robot_base must describe the URDF base link. Legacy rollouts can infer the source translation from --reference and use configured identity rotation; this is printed as an inference, not a measured pose.

The target hand is computed as:

```
T_CRXbase_hand = inverse(T_targetScene_CRXbase)
                 @ T_targetScene_sourceEnv
                 @ T_sourceEnv_RBYbase
                 @ FK_RBY(q_arm)
```

--scene-transform is T_targetScene_sourceEnv. Apply one common scene transform to the entire task, including both hands and objects. Transformed object trajectories are saved in the NPZ, alongside diagnostics and transform values.

Bimanual mode requires --left-base and --right-base in the same target scene frame. For a single arm the selected base defaults to identity. Independent single-arm placement searches must not be combined as a coordinated bimanual trajectory.

## Placement, timing and bounds

--search-placement currently supports one side. It evaluates nine placements around configured initial CRX configurations on 12 sampled frames, then solves the entire trajectory at the selected placement. It optimizes geometric residuals only. A general scene rotation also rotates the source table plane: a reachable placement is not a physical tabletop setup recommendation.

--time-scale scales duration, not geometry. Reports include joint speed/URDF-limit ratios and arm acceleration. These limits do not certify real-controller feasibility. --clip-hand-tolerance is an explicit upper bound on permitted finger limit corrections in radians; 0 preserves raw values and reports violations. If enabled, corrections greater than the supplied tolerance raise an error. Original H5 files are not edited.

The inactive side in qpos56 uses configured initial values, so single-arm output must not be sent directly to a bimanual backend.

## Local hammer validation

Legacy rollout: 358 frames at 60 Hz. Reference: 749 frames at 120 Hz, source base translation [0,0,-0.85]. Final rollout qpos exactly matches reference frame 0; removing it leaves 357 frames.

Right-arm scene search converges for all 357 frames without arm limit hits. At the tested placement, maximum model residual is approximately 1.47e-10 m / 2.87e-10 rad. This is numerical model agreement, not physical accuracy. Raw finger angles exceed matching source/target limits by at most 0.00413934 rad. Explicit 0.005 rad tolerance clips 990 sampled values. With 2x duration, maximum joint-speed/URDF-limit ratio is 0.84315; maximum arm acceleration remains approximately 72.22 rad/s^2.

Actual CRX installation transforms, collisions, table/strike location, contact dynamics, initial transition and hardware acceleration limits remain unverified. ConTrack export changes must also be exercised on the Isaac Lab server; local checks are offline only.

## Tests

```bash
env -u PYTHONPATH .venv/bin/python -m pytest tests/test_h5_replay_review.py -q
```


## Horizontal bimanual tabletop search

Use --side both --search-tabletop --table-height -0.195 with explicit --left-base and --right-base. The measured operation surface is approximately 19.5 cm below the left robot mounting plane. The source plane is assumed z=0; override --source-table-z if the source support plane differs. This maps support planes, not the object origin directly onto the table.

Only xy translation and yaw vary; roll/pitch remain zero. --anchor-object defaults to object_0. --table-x-range MIN MAX COUNT and --table-y-range MIN MAX COUNT bound the initial object-origin xy positions in the shared scene, not measured table boundaries. Defaults are x=[0.35,0.65], y=[-0.55,-0.25], three samples each. --table-yaws-deg defaults to 0 90 180 270. These are exploratory search bounds; table extent, hand/arm/object collision and support geometry are not checked.

The search screens three frames, then checks 12 frames, then all frames. A candidate is accepted only if both arms converge on all frames without near-limit solutions. IK branch choices and finite sampling of placements can cause false negatives. Failure in this finite grid does not prove physical unreachability. Successful output is still offline only and requires fresh velocity/acceleration/jerk checks, transition planning and collision validation. The output records table_height/source_table_z and the chosen scene transform. Existing --search-placement behavior is unchanged.


## Finite tabletop clearance screening

scripts/check_h5_table_clearance.py transforms all available URDF collision STL triangles and reference object mesh triangles at every saved trajectory frame. STL scales, collision origins and full finger joints are included. Triangles are clipped to the finite xy tabletop footprint before computing minimum height, so triangles crossing table edges are not missed merely because all vertices lie outside. No new dependencies or ROS are required. Missing/unsupported meshes fail explicitly; links without collision geometry are listed in the report.

Measured approximate table bounds in the shared left-base frame: x=[0.18358936,0.88358936], y=[-1.15578132,0.24421868], operation surface z=-0.195 meters. Bounds assume +X points toward the table, table centered on the two base centers, and the right base center is 0.153m behind the rear edge. The 0.0087m base-height calibration difference and tape-measure uncertainty remain.

```bash
.venv/bin/python scripts/check_h5_table_clearance.py --trajectory outputs/hammer/both_tabletop_slow7.npz --reference-h5 ~/data/grab-s1_hammer_use_1-rby1a_sharpa.h5 --object-map object_0=0 --table 0.18358936 0.88358936 -1.15578132 0.24421868 -0.195 --out outputs/hammer/table_clearance.json
```

The 2mm default tolerance is only a reporting threshold, not clearance certification. The tested forbidden region extends downward indefinitely within the table footprint, conservatively covering under-table intrusion. This is not a watertight mesh/solid intersection engine. It does not check self/inter-arm/object-robot collisions, unmodeled stands, inside-volume containment cases, between-frame swept motion, initial/final transitions or physical calibration error. Zero flagged entities must never be called collision-free. Object support contact near zero clearance is reported without automatically classifying it as an invalid task contact.


## Passive browser preview

scripts/view_h5_replay.py consumes qpos56 directly and displays both arms/hands using their collision meshes, the recorded transformed hammer mesh, and a finite table. It never imports ROS or sends commands. No physics/collision detection is performed. The table display is 3cm thick for visibility only, not a measured underside. Playback starts paused, supports frame scrubbing, speed control and optional looping. Localhost binding is default. If accessing a remote machine, use SSH port forwarding (ssh -L 9219:127.0.0.1:9219 user@host), then open http://127.0.0.1:9219 on your own computer. This uses the existing optional viser dependency; install it only if missing.

```bash
.venv/bin/python scripts/view_h5_replay.py --trajectory outputs/hammer/both_tabletop_slow7.npz --reference-h5 ~/data/grab-s1_hammer_use_1-rby1a_sharpa.h5 --table 0.18358936 0.88358936 -1.15578132 0.24421868 -0.195
```

--validate-only checks files/meshes without importing viser or starting a server.


## Software-mock execution (not hardware replay)

New files: scripts/run_h5_mock.py, h5_mock_guard.py, h5_mock_playback.py. run_h5_replay.py remains offline with --execute disabled. Default invocation of the mock runner only validates the NPZ, joint bounds, source velocities and complete bimanual IK diagnostics. --run-mock explicitly enables publication. No dependency additions or changes to the existing backend are made.

Mock execution requires a dedicated ROS_DOMAIN_ID=187 and ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST in both terminals. Do not use the physical-device domain (previously 87). The runner queries both controller managers through ListHardwareComponents, requiring exclusively active mock_components/GenericSystem plugins, and both hand_node parameters requiring backend=mock/read_only=false. Missing services, ambiguous node identities, competing publishers, stale verification and physical plugins fail closed. Verification continues while running. These are operational checks, not a security boundary against a malicious ROS participant.

First stop the previous mock launch with Ctrl+C, then open a mock-only launch terminal:

```bash
source /opt/ros/jazzy/setup.bash
source ~/ws_fanuc/install/setup.bash
export ROS_DOMAIN_ID=187
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
ros2 launch bimanual_manipulation bimanual_system.launch.py crx_mock:=true sharpa_backend:=mock rviz:=true viewer:=false method:=linear input_rate_hz:=50.0
```

In a separate terminal:

```bash
source /opt/ros/jazzy/setup.bash
source ~/ws_fanuc/install/setup.bash
export ROS_DOMAIN_ID=187
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
cd ~/retargeting_crx_sharpa
.venv/bin/python -c "import rclpy, sensor_msgs.msg, controller_manager_msgs.srv; print('ROS imports OK')"
.venv/bin/python scripts/run_h5_mock.py --trajectory outputs/hammer/both_tabletop_slow7.npz
.venv/bin/python scripts/run_h5_mock.py --trajectory outputs/hammer/both_tabletop_slow7.npz --run-mock
```

Do not unset PYTHONPATH/LD_LIBRARY_PATH for ROS execution; the interpreter must be ROS Jazzy compatible. If imports fail, stop and fix the Python/ROS environment rather than installing guessed pip ROS packages.

The runner samples commands at 50Hz, matching the explicit linear interpolation input rate, preserving the saved trajectory duration. Initial and resumed entry use a quintic ease from measured feedback, with at least a 5s transition extended based on velocity and experimental arm acceleration/jerk budgets. This does not collision-check the entry or certify hardware behavior. Ordinary playback is linear resampling, so smooth acceleration across saved waypoint corners is not guaranteed. The isolated mock test intentionally verifies message routing/tracking, not physics or contact success.

Controls: p + Enter pauses and requests one measured-pose hold through the existing backend; r + Enter re-enters the paused trajectory point from fresh measured feedback before continuing; q + Enter or Ctrl+C stops. EOF also stops. Final target continues until feedback settles within 0.02rad arms / 0.05rad hands for 0.5s, timeout 10s. Stale feedback, lost mock checks, missed deadlines >0.2s and tracking error >0.5rad stop playback. CSV records elapsed time, phase, trajectory progress, target and measured positions. Closing calls the backend's measured hold when valid feedback is available; missing feedback means cease publishing and leave driver timeout behavior in charge. Process death and arbitrary middleware failures cannot guarantee a hold. No real e-stop claim is implied.

The mock test does NOT display a physically attached hammer in RViz: only robot joints are sent to the driver. Use the passive preview for recorded object motion. Do not take this as evidence that a real hand can grip or strike. Real-device execution remains unsupported. Locally only ROS-free tests and validation can run; installed ROS/Jazzy integration must be checked on the server.


## Separate single-arm empty-hand commissioning test

New scripts: run_h5_arm_smoke.py and h5_arm_smoke.py. The existing full bimanual mock runner remains mock-only; no guard has been bypassed. The new test does NOT execute the complete hammer trajectory. It takes 30 source arm frames (configurable 6..60), smooths a relative 6-joint path, reduces its largest excursion to at most approximately 1 degree, anchors it to the measured current arm pose, traverses slowly and reverses back to that pose. Normal end returns; interruption does not automatically return. No finger or inactive-arm commands are created. Playback carries no hammer/object target.

Default --mode prepare only writes relative_joints to an NPZ, without ROS imports/connections:

```bash
.venv/bin/python scripts/run_h5_arm_smoke.py --trajectory outputs/hammer/both_tabletop_slow7.npz --side right
```

First test this new code on software mock, with the same isolated domain 187, LOCALHOST discovery and method:=linear input_rate_hz:=50.0 launch used above:

```bash
.venv/bin/python scripts/run_h5_arm_smoke.py --trajectory outputs/hammer/both_tabletop_slow7.npz --side right --mode mock --out outputs/hammer/right_smoke_mock.npz
```

Live modes first read feedback and verify the selected controller manager's hardware plugin, active forward controller, interpolation method/rate, unique graph identities and absence of competing command senders. Mock expects active mock_components/GenericSystem; real expects active fanuc_robot_driver/FanucHardwareInterface. The absolute path is saved in *_anchored.npz and current/planned joint ranges are printed before explicit operator confirmation. Start-pose drift aborts rather than silently re-anchoring. --mode real is an explicit, distinct physical path requiring domain 87 and LOCALHOST. It must NOT be run before the new mock test and a local clear-space review; no physical test was run during development.

Real use requires the local operator to place the selected arm empty-handed in a clear elevated pose, verify the entire small path is clear of the table/other arm/cables, and have the robot's actual stop control available. The tool does not automatically certify Cartesian or inter-arm clearance, safe stopping, actual hand mounting, payload, or human clearance. Small joint motion can still cause contact near an obstacle or singularity. Keep other senders stopped, and remember driver startup may enable devices independently of this script. Do not launch physical Sharpa just to test arm communication. Use a selected-arm driver launch where possible and review actual device/stop configuration on site.

When this review is complete, the physical command uses the same arguments with --mode real, followed by literal START right real (or left) at the displayed confirmation. It sends only complete selected-side J1..J6 JointState commands to /crx5ia/joint_targets at 50Hz, and reads /crx5ia/SIDE/joint_states. No hardware test was performed locally. Stop with q + Enter or Ctrl+C; this requests one measured hold if feedback is fresh, otherwise ceases publishing and relies on driver behavior. This is not an emergency stop. Missed deadlines, stale/frozen feedback, >0.03rad tracking error, position-margin violations, or runtime command-speed/excursion cap violations abort. CSV records target/feedback. Successful completion requires feedback to settle near the original start.

The smoothed path is conservatively time-scaled based on dense derivative samples (planning budgets velocity .008rad/s, acceleration .008rad/s2, jerk .02rad/s3); it is not an analytic manufacturer guarantee and downstream linear interpolation can alter derivatives. The initial local right-arm preparation has max excursion approximately 0.98 degrees and total duration 48.32s. These scripts must be validated on installed ROS/Jazzy; only headless pure planning tests are available locally.


## Explicit tabletop physical replay (new entry point)

run_h5_real.py and h5_real_guard.py are separate from the mock-only runner. Default invocation writes an offline review to outputs/hammer/real_review/plan.json, prints object placement and does not import ROS. It uses load_motion from the mock script solely as a ROS-free data validator. Hardware mode was confirmed by the user, who chose hammer-on-table initial grasp and deferred the known near-left-hand/table issues. Those issues remain unresolved; this implementation is not a safety/collision approval or proof of grasp success.

```bash
.venv/bin/python scripts/run_h5_real.py --trajectory outputs/hammer/both_tabletop_slow7.npz --reference-h5 ~/data/grab-s1_hammer_use_1-rby1a_sharpa.h5
```

Additional first-run time scaling applies conservative joint planning budgets (.2rad/s arm, .4rad/s hand, .5rad/s2 arm, 2rad/s3 arm) over the source samples with 10% duration margin. This is additional to the saved 7x scaling, not a change to NPZ. Local output duration is approximately 239.3s plus initial entry; this slowdown substantially changes contact dynamics. Targets are still linearly resampled at 50Hz, so finite-difference derivative limits are not continuous hardware guarantees.

Physical launch must be the confirmed bimanual system with actual arm/hand backends, method:=linear input_rate_hz:=50.0, ROS_DOMAIN_ID=87 and ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST. Launching physical Sharpa can enable fingers before replay runs. RealGuard requires active fanuc_robot_driver/FanucHardwareInterface components, active forward controllers, sharpa_sdk hands in writable MIT/position mode, matching interpolation rate, unique node identities, no competing publishers and continuously refreshed verification. Missing or mismatched checks refuse execution.

Adding --run-real opens read/verification connections and waits for literal START REAL TABLETOP REPLAY after showing measured joints, first targets and entry duration. No trajectory command is sent before confirmation. Cancel-before-first-command uses backend.close(send_hold=False) to avoid publishing even a hold; suppression is prohibited after commands were accepted. Entry starts from live feedback, uses at least 15s with conservative joint rates, but does not collision-check the entry. The plan is NOT re-anchored after approval if joints drift by more than .01rad.

Hammer must be placed to match the saved object first pose, scale and orientation in the shared scene frame. Current object origin approximately [0.65,-0.4,-0.183392]m is not a directly measurable corner/grasp point. plan.json includes mesh bounds and orientation for alignment; use the passive viewer and actual geometry, not only the origin point. There is no perception, grasp detector or force control added. Start with empty hands: the recorded fingers open/close during approach and grasp.

q+Enter/Ctrl+C stops; p+Enter requests measured hold. Physical resume is deliberately disabled after any pause because object state may have changed. Releasing/holding an actual tool is not guaranteed by software shutdown or process death; hardware stop procedures and the responsible on-site operator remain necessary. Joint tracking thresholds stop on large lag/contact instead of driving farther to force a grasp. End waits for joint settling but never reports physical task success. This entry has not been executed on actual hardware during development; default-only offline preparation and headless tests are the locally verified scope.


## Preview the actual measured-to-first entry before physical replay

prepare_h5_entry_preview.py reads the exact reviewed_start.npz saved by run_h5_real.py, checks it matches the supplied source trajectory, and generates the quintic joint interpolation used for entry. Objects stay fixed at their first recorded table pose. It sends no commands and cannot certify clearance. Output is tagged preview_only and lacks execution diagnostics; the mock/physical motion loader and small-arm entry explicitly reject it.

```bash
.venv/bin/python scripts/prepare_h5_entry_preview.py --trajectory outputs/hammer/both_tabletop_z175_slow7.npz --reviewed-start outputs/hammer/real_review_z175/reviewed_start.npz --out outputs/hammer/entry_z175_preview.npz
.venv/bin/python scripts/view_h5_replay.py --trajectory outputs/hammer/entry_z175_preview.npz --reference-h5 ~/data/grab-s1_hammer_use_1-rby1a_sharpa.h5 --table 0.18358936 0.88358936 -1.15578132 0.24421868 -0.175 --port 9220
```

Open http://127.0.0.1:9220 and inspect the entire approach using the frame slider. The prior example has right-arm J1/J4 changes around 66/67 degrees, unlike the 1-degree smoke motion. A preview without visible collision does not approve the physical path. If hardware moves after saving reviewed_start, this preview is stale; generate a new start snapshot without approving motion. The 2cm pad footprint and all obstacles also need to match the displayed scene.


For one continuous display-only entry + task replay, add --include-replay to prepare_h5_entry_preview.py. It preserves the reviewed entry, then linearly resamples recorded joint motion and interpolates object rotations with SLERP at run_h5_real's conservative extra slowdown. entry_end_frame identifies the boundary. The object is fixed during entry and independently follows its recorded poses afterwards; visible grasp is not proof of physical grasp. This file remains explicitly non-executable. Existing entry-only behavior is unchanged.

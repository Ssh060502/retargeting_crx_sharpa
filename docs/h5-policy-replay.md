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

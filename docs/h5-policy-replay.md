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

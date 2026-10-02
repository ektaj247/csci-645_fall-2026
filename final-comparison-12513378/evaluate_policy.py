"""Fixed-condition diagnostic evaluation for the migrated CSCI645 task.

Each environment contributes exactly its first complete episode. Terminal data
are read before explicit resets. No learning or checkpoint writes occur.
"""
import argparse
import csv
import hashlib
import inspect
import json
import os
from pathlib import Path
import runpy
import statistics
import sys

ROOT = Path('/project2/seita_1951/jichkar/csci645')
DEFAULT_REPO = ROOT / 'csci645-mjlab160'
TASK = 'Mjlab-Leap-Left-HandCube-Rotate'
RUN = 'logs/rsl_rl/leap_left_hand_cube_rotate/2026-09-28_01-39-33_baseline_mjlab160_seed0'

def summarize(rows):
    result = {'episodes': len(rows)}
    fields = ['duration_s', 'rotation_progress', 'position_error_m', 'tilt_error_rad',
              'linear_speed_m_s', 'contact_fraction', 'signed_yaw_rad',
              'mean_torque_squared_Nm2', 'mean_abs_joint_power_W',
              'sampled_abs_work_J', 'action_clip_fraction']
    for key in fields:
        values = [float(r[key]) for r in rows]
        result[key] = {'mean': statistics.mean(values),
                       'std_across_episodes': statistics.stdev(values) if len(values)>1 else 0.0}
    for key in ['failed', 'cube_fell', 'cube_pose_deviation', 'cube_too_fast', 'nan', 'time_out']:
        result[key] = {'count': sum(int(r[key]) for r in rows),
                       'fraction': sum(int(r[key]) for r in rows)/len(rows)}
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, default=DEFAULT_REPO)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--mass', choices=['standard', 'light', 'heavy'], default='standard')
    parser.add_argument('--mode', choices=['deterministic', 'sampled'], default='deterministic')
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    assert args.episodes > 0
    repo = args.repo.resolve()
    assert repo.is_relative_to(ROOT), 'Use the project2 repository'
    checkpoint = (args.checkpoint or repo / RUN / 'model_4999.pt').resolve()
    assert checkpoint.is_file(), checkpoint
    os.chdir(repo)
    sys.path.insert(0, str(repo / 'scripts'))
    os.environ.setdefault('WANDB_MODE', 'offline')
    os.environ.setdefault('MUJOCO_GL', 'glfw' if args.preflight else 'egl')

    import torch
    from tensordict import TensorDict
    from rsl_rl.models import MLPModel
    from rsl_rl.runners import OnPolicyRunner
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
    from mjlab.utils.lab_api.math import euler_xyz_from_quat, wrap_to_pi
    import in_hand_rotation_mjlab
    import in_hand_rotation_mjlab.tasks

    assert Path(in_hand_rotation_mjlab.__file__).resolve().is_relative_to(repo)
    assert 'stochastic_output' in inspect.signature(MLPModel.forward).parameters, (
        'Unexpected MLP forward signature: ' + str(inspect.signature(MLPModel.forward)))
    training = runpy.run_path(str(repo / 'scripts/train.py'), run_name='evaluation_helpers')
    agent = load_rl_cfg(TASK)
    prepared = training['_prepare_agent_cfg'](agent)
    cfg = load_env_cfg(TASK, play=False)
    assert abs(cfg.episode_length_s - 20.0) < 1e-9
    assert cfg.observations['actor'].enable_corruption
    assert tuple(cfg.events['dr_cube_mass'].params['mass_range']) == (0.7, 1.4)

    if args.preflight:
        import copy
        obs = TensorDict({'actor': torch.zeros(2, 320), 'critic': torch.zeros(2, 83)}, batch_size=[2])
        model_cfg = copy.deepcopy(prepared['actor'])
        assert model_cfg.pop('class_name') == 'MLPModel'
        model = MLPModel(obs, prepared['obs_groups'], 'actor', 16, **model_cfg).eval()
        with torch.inference_mode():
            for sampled in [False, True]:
                actions = model(obs, stochastic_output=sampled)
                assert actions.shape == (2, 16) and torch.isfinite(actions).all()
        print('EVALUATION PREFLIGHT PASSED: imports, model modes, dimensions, horizon, noise, mass range')
        print('Checkpoint:', checkpoint)
        return

    assert torch.cuda.is_available(), 'Submit evaluation through Slurm with a GPU'
    assert args.output is not None, '--output is required for GPU evaluation'
    output = args.output.resolve()
    assert output.is_relative_to(ROOT), 'Save evaluation results in project2'
    output.mkdir(parents=True, exist_ok=False)
    cfg.scene.num_envs = args.episodes
    cfg.seed = args.seed
    cfg.auto_reset = False
    # Rewards do not drive inference; prevent curriculum updates during scoring.
    cfg.curriculum = {}
    mass_range = {'standard': (0.7, 1.4), 'light': (0.7, 0.7), 'heavy': (1.4, 1.4)}[args.mass]
    cfg.events['dr_cube_mass'].params['mass_range'] = mass_range
    protocol = {
        'purpose': 'baseline diagnosis before choosing modifications',
        'task': TASK, 'checkpoint': str(checkpoint),
        'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'episodes': args.episodes, 'seed': args.seed, 'mode': args.mode,
        'episode_horizon_s': cfg.episode_length_s,
        'mass_multiplier_range': mass_range,
        'actor_noise': True, 'other_randomization': 'baseline defaults',
        'curriculum': 'disabled during evaluation only',
        'aggregation': 'one first complete episode per environment; equal episode weighting',
        'video_selection': 'environment 0 first episode, selected before observation of results',
        'rotation_progress': 'repository MetricsManager score, dimensionless, not rotations',
        'signed_yaw_rad': 'negative sum of wrapped yaw differences from post-forward poses',
        'effort': 'control-step samples; torque squared summed over actuators; absolute joint power summed over joints',
        'failure': 'any non-timeout termination, including simultaneous timeout and failure',
        'mass_stress': 'endpoints of the baseline training range, not unseen masses',
    }
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    device = 'cuda:0'
    env = None
    writer = None
    rows = []
    try:
        env = ManagerBasedRlEnv(cfg=cfg, device=device, render_mode='rgb_array' if args.video else None)
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
        runner = OnPolicyRunner(wrapped, prepared, log_dir=None, device=device)
        runner.load(str(checkpoint), map_location=device)
        policy = runner.get_inference_policy(device=device)
        assert not policy.training
        obs_dict, _ = env.reset(seed=args.seed)
        obs = TensorDict(obs_dict, batch_size=[args.episodes])
        cube = env.scene['cube']
        robot = env.scene['robot']
        prev_yaw = euler_xyz_from_quat(cube.data.root_link_quat_w)[2].clone()
        active = torch.ones(args.episodes, dtype=torch.bool, device=device)
        steps = torch.zeros(args.episodes, dtype=torch.long, device=device)
        metric_names = list(env.metrics_manager.active_terms)
        mapping = {'rotation_progress':'rotation_progress', 'position_error_m':'position_error',
                   'tilt_error_rad':'tilt_error', 'linear_speed_m_s':'linear_speed',
                   'contact_fraction':'fingertip_contact_fraction'}
        for name in mapping.values(): assert name in metric_names, name
        totals = {k: torch.zeros(args.episodes, device=device) for k in mapping}
        for key in ['signed_yaw_rad','mean_torque_squared_Nm2','mean_abs_joint_power_W','action_clip_fraction']:
            totals[key] = torch.zeros(args.episodes, device=device)
        dt = env.step_dt
        protocol['control_step_s'] = dt
        if args.video:
            import imageio.v2 as imageio
            writer = imageio.get_writer(str(output / 'episode_000.mp4'), fps=round(1/dt))
            writer.append_data(env.render())

        with torch.inference_mode():
            for step in range(env.max_episode_length + 1):
                actions = policy(obs, stochastic_output=args.mode == 'sampled')
                assert torch.isfinite(actions).all(), 'Policy produced nonfinite actions'
                clipped = (actions.abs() > agent.clip_actions).float().mean(dim=-1)
                next_obs, _, dones, _ = wrapped.step(actions)
                # auto_reset=False keeps terminal state and metric buffers intact here.
                for key, name in mapping.items():
                    values = env.metrics_manager._step_values[:, metric_names.index(name)]
                    assert torch.isfinite(values[active]).all(), f'Nonfinite metric: {name}'
                    totals[key][active] += values[active]
                yaw = euler_xyz_from_quat(cube.data.root_link_quat_w)[2]
                tau = robot.data.actuator_force
                velocity = robot.data.joint_vel
                assert tau.shape == velocity.shape and tau.shape[1] == 16
                assert torch.isfinite(tau[active]).all() and torch.isfinite(velocity[active]).all()
                totals['signed_yaw_rad'][active] -= wrap_to_pi(yaw-prev_yaw)[active]
                totals['mean_torque_squared_Nm2'][active] += tau.square().sum(-1)[active]
                totals['mean_abs_joint_power_W'][active] += (tau*velocity).abs().sum(-1)[active]
                totals['action_clip_fraction'][active] += clipped[active]
                steps[active] += 1
                prev_yaw.copy_(yaw)
                if writer is not None and bool(active[0]): writer.append_data(env.render())
                finished = dones.bool() & active
                for i in finished.nonzero().flatten().tolist():
                    length = int(steps[i])
                    row = {'episode_id': i, 'steps': length, 'duration_s': length*dt,
                           'failed': bool(env.termination_manager.terminated[i])}
                    for name in ['cube_fell','cube_pose_deviation','cube_too_fast','nan','time_out']:
                        assert name in env.termination_manager.active_terms
                        row[name] = bool(env.termination_manager.get_term(name)[i])
                    for key, tensor in totals.items():
                        row[key] = float(tensor[i]) / (1 if key == 'signed_yaw_rad' else length)
                    row['sampled_abs_work_J'] = row['mean_abs_joint_power_W']*row['duration_s']
                    rows.append(row)
                active[finished] = False
                if not active.any(): break
                reset_ids = dones.nonzero().flatten()
                if len(reset_ids):
                    # Later episodes in these environments never contribute results.
                    reset_obs, _ = env.reset(env_ids=reset_ids)
                    # Keep the already-returned observations for still-active envs.
                    for key in reset_obs:
                        next_obs[key][reset_ids] = reset_obs[key][reset_ids]
                    prev_yaw[reset_ids] = euler_xyz_from_quat(cube.data.root_link_quat_w)[2][reset_ids]
                obs = next_obs
                if (step+1) % 100 == 0:
                    print(f'Control step {step+1}: {len(rows)}/{args.episodes} episodes finished', flush=True)
        assert len(rows) == args.episodes, f'Incomplete episodes: {len(rows)}'
        rows.sort(key=lambda r:r['episode_id'])
        with (output / 'episodes.csv').open('w', newline='') as f:
            writer_csv = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer_csv.writeheader(); writer_csv.writerows(rows)
        results = {'protocol': protocol, 'summary': summarize(rows)}
        (output / 'results.json').write_text(json.dumps(results, indent=2, allow_nan=False))
        print(json.dumps(results['summary'], indent=2), flush=True)
        print('EVALUATION COMPLETE:', output, flush=True)
    finally:
        if writer is not None: writer.close()
        if env is not None: env.close()

if __name__ == '__main__':
    main()

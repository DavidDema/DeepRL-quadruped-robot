import numpy as np
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium.spaces import Box
import os
import torch


class GOEnv(MujocoEnv):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 25,
    }

    class Cfg:
        class RewardScale:
            termination = -0.0
            tracking_lin_vel = 2.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            #orientation = -0.
            #torques = -0.00002
            #dof_vel = -0.
            dof_acc = -2.5e-7
            #base_height = -0.
            action_rate = -0.01

        class Control:
            control_type = 'P'
            stiffness = 3.0 # 10.0
            damping = 0.5 # 2.0
            action_scale = 0.5

        def __init__(self):
            self.base_height_target = 0.6
            self.reward_scale = self.RewardScale()
            self.control = self.Control()
            self.commands = [0.5, 0.0, 0.0] # x, y, yaw
            self.tracking_sigma = 0.25

    def __init__(self,
                 healthy_z_range=(0.15, 0.5),
                 reset_noise_scale=1e-2,
                 terminate_when_unhealthy=True,
                 exclude_current_positions_from_observation=False,
                 frame_skip=40,
                 **kwargs,
                 ):
        if exclude_current_positions_from_observation:
            self.obs_dim = 17 + 18
        else:
            self.obs_dim = 19 + 18

        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float64
        )
        MujocoEnv.__init__(self,
                           model_path=os.path.join(os.path.dirname(__file__), 'go/scene.xml'),
                           frame_skip=frame_skip,
                           observation_space=observation_space,
                           **kwargs
                           )
        self.action_dim = 12
        self.action_space = Box(
            low=self.lower_limits, high=self.upper_limits, shape=(self.action_dim,), dtype=np.float64
        )

        self._reset_noise_scale = reset_noise_scale
        self._healthy_z_range = healthy_z_range
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._exclude_current_positions_from_observation = exclude_current_positions_from_observation

        self.cfg = self.Cfg()

        self.action = np.zeros(12)
        self.last_action = np.zeros(12)

        self.base_pos = self.init_joints[:3]
        self.base_lin_vel = np.zeros(3)
        self.base_ang_vel = np.zeros(3)
        self.base_orientation = np.zeros(3)
        self.dof_pos = self.init_joints[-12:]
        self.dof_vel = np.zeros(12)
        self.last_base_pos = self.init_joints[:3]
        self.last_base_vel = np.zeros(3)
        self.last_base_orientation = np.zeros(3)
        self.last_dof_pos = self.init_joints[-12:]
        self.last_dof_vel = np.zeros(12)

        #self.p_gain = torch.zeros(self.action_dim, dtype=torch.float, requires_grad=False) * self.cfg.control.stiffness
        #self.d_gain = torch.zeros(self.action_dim, dtype=torch.float, requires_grad=False) * self.cfg.control.damping

        # torques for simulation
        self.p_gains = 2.0
        self.d_gains = 0.2

    @property
    def lower_limits(self):
        #return np.array([-0.863, -0.686, -2.818]*4)
        return np.array([-0.2, -1.0, -1.2] * 4)

    @property
    def upper_limits(self):
        #return np.array([0.863, 4.501, -0.888]*4)
        return np.array([0.2, 1.0, -0.9] * 4)

    @property
    def init_joints(self):
        #return np.array([0, 0, 0.37, 1, 0, 0, 0] + [0, 0.7, -1.4] * 4)
        return np.array([0, 0, 0.368, 1, 0, 0, 0] + [0, 0.6, -1.6]*4)

    @property
    def base_rotation(self):
        """
        compute root (base) rotation of the robot. The rotation can be used for rewards
        :return: rotation of root in xyz direction
        """
        q = self.data.qpos[3:7].copy()
        q0 = q[0]
        q1 = q[1]
        q2 = q[2]
        q3 = q[3]

        x = np.arctan2(2 * (q0 * q1 - q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        y = np.arcsin(np.clip(2 * (q1 * q3 + q0 * q2), -1, 1))
        z = np.arctan2(2 * (q0 * q3 - q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))

        return np.array([x, y, z])

    @property
    def is_healthy(self):
        min_z, max_z = self._healthy_z_range
        is_healthy = min_z < self.data.qpos[2] < max_z

        return is_healthy

    @property
    def terminated(self):
        terminated = (not self.is_healthy) and self._terminate_when_unhealthy
        return terminated

    def _get_obs(self):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()

        if self._exclude_current_positions_from_observation:
            qpos = qpos[2:]

        return np.concatenate([qpos, qvel])

    # ------------ reward functions----------------
    def _reward_healthy(self):
        return (self.is_healthy - 1)

    '''
    def _reward_base_height(self):
        # Penalize base height away from target
        return np.square(self.base_pos[2] - self.cfg.base_height_target)
    '''
    '''
    def _reward_lin_vel(self, before_pos, after_pos, k=10.0):
        target_vel = np.array([0.5, 0, 0])
        lin_vel = (after_pos - before_pos) / self.dt
        #return np.exp(-k*np.linalg.norm(target_vel - lin_vel)/self.cfg.tracking_sigma)
        return np.exp(-k * np.linalg.norm(target_vel - lin_vel))
    '''
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = np.sum(np.square(self.cfg.commands[:2] - self.base_lin_vel[:2]))
        return np.exp(-lin_vel_error / self.cfg.tracking_sigma)

    def _reward_tracking_ang_vel(self): # tracking_ang_vel = 0.5
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = np.square(self.cfg.commands[2] - self.base_ang_vel[2])
        return np.exp(-ang_vel_error/self.cfg.tracking_sigma)
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return np.square(self.base_lin_vel[2])

    def _reward_ang_vel_xy(self): # ang_vel_xy = -0.05
        # Penalize non flat base orientation
        return np.sum(np.square(self.base_ang_vel[2]))

    def _reward_dof_acc(self): # dof_acc = -2.5e-7
        # Penalize dof accelerations
        return np.sum(np.square((self.last_dof_vel - self.dof_vel) / self.dt))

    '''
    def _reward_torques(self): # torques = -0.00001
        # Penalize torques
        return np.sum(np.square(self.torques))
    '''

    def _reward_action_rate(self): # action_rate = -0.01
        # Penalize changes in actions
        return np.sum(np.square(self.last_action - self.action))

    # ------ Reward Stefan ------
    def _reward_going_far_x(self):
        reward_x = self.data.qpos[0]
        return reward_x

    def _reward_going_far_y(self):
        reward_y = self.data.qpos[1]
        return -np.abs(reward_y)

    # ---------------------------
    def step(self, delta_q, torque_sim=True):
        action = delta_q + self.data.qpos[-12:]
        action = np.clip(action, a_min=self.lower_limits, a_max=self.upper_limits)

        #before_pos = self.data.qpos[:3].copy()
        self_last_base_pos = self.data.qpos[:3].copy()
        self.last_base_vel = self.data.qvel[:3].copy()
        self.last_base_orientation = self.base_rotation
        self.last_dof_pos = self.data.qpos[-12:]
        #self.last_dof_vel = (self.last_dof_pos - self.dof_pos) / self.dt
        self.dof_vel = self.data.qvel[-12:]

        if torque_sim:
            torque_sim_param = self.p_gains * delta_q - self.d_gains * self.data.qvel[-12:]
            action = np.clip(torque_sim_param, a_min=self.lower_limits, a_max=self.upper_limits)
            self.do_simulation(action, self.frame_skip)
        else:
            self.do_simulation(action, self.frame_skip)

        #after_pos = self.data.qpos[:3].copy()
        self.base_pos = self.data.qpos[:3].copy()
        self.base_lin_vel = self.data.qvel[:3].copy()
        self.base_orientation = self.base_rotation
        self.base_ang_vel = (self.last_base_orientation - self.base_orientation) / self.dt
        self.dof_pos = self.data.qpos[-12:]
        #self.dof_vel = (self.last_dof_pos - self.dof_pos) / self.dt
        self.dof_vel = self.data.qvel[-12:]

        #lin_v_track_reward = self._reward_lin_vel(before_pos, after_pos, k=10.0)
        healthy_reward = self._reward_healthy()
        reward_tracking_lin_vel = self._reward_tracking_lin_vel()
        reward_tracking_ang_vel = self._reward_tracking_ang_vel()
        reward_lin_vel_z = self._reward_lin_vel_z()
        reward_ang_vel_xy = self._reward_ang_vel_xy()
        reward_dof_acc = self._reward_dof_acc()
        reward_action_rate = self._reward_action_rate()
        reward_going_far_x = self._reward_going_far_x()
        reward_going_far_y = self._reward_going_far_y()

        total_rewards = np.sum([
            healthy_reward,
            # 5 * lin_v_track_reward,
            self.cfg.reward_scale.tracking_lin_vel * reward_tracking_lin_vel,
            self.cfg.reward_scale.tracking_ang_vel * reward_tracking_ang_vel,
            self.cfg.reward_scale.lin_vel_z * reward_lin_vel_z,
            self.cfg.reward_scale.ang_vel_xy * reward_ang_vel_xy,
            self.cfg.reward_scale.dof_acc * reward_dof_acc,
            self.cfg.reward_scale.action_rate * reward_action_rate,
            1.0 * reward_going_far_x,
            1.0 * reward_going_far_y
        ])

        terminate = self.terminated
        observation = self._get_obs()
        info = {
            'total_reward': total_rewards,
            "healthy_reward": healthy_reward,
            #'lin_v_track_reward': lin_v_track_reward,
            'reward_tracking_lin_vel': reward_tracking_lin_vel,
            'reward_tracking_ang_vel': reward_tracking_ang_vel,
            'reward_lin_vel_z': reward_lin_vel_z,
            'reward_ang_vel_xy': reward_ang_vel_xy,
            'reward_dof_acc': reward_dof_acc,
            'reward_action_rate': reward_action_rate,
            'reward_going_far_x': reward_going_far_x,
            'reward_going_far_y': reward_going_far_y,
            "traverse": self.data.qpos[0],
        }

        if self.render_mode == "human":
            self.render()
        return observation, total_rewards, terminate, info

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_joints + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )
        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

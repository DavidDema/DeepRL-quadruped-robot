import numpy as np
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium.spaces import Box
import os


class GOEnv(MujocoEnv):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 25,
    }

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

        self.timestep = 0
        self.max_timesteps = 1

    @property
    def lower_limits(self):
        return np.array([-0.863, -0.686, -2.818]*4)

    @property
    def upper_limits(self):
        return np.array([0.863, 4.501, -0.888]*4)

    @property
    def init_joints(self):
        # base position (x, y, z), base orientation (quaternion), 4x leg position (3 joints) 
        return np.array([0, 0, 0.37, 1, 0, 0, 0] + [0, 0.7, -1.4]*4)

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
        is_healthy_z = min_z < self.data.qpos[2] < max_z

        # convert orientation from radiant to degree
        orientation = 360 * np.array(self.base_rotation) / (2 * np.pi)

        # Angles over "unhealthy_angle" degree are unhealthy
        unhealthy_angle = 300
        is_healthy_pitch = np.abs(orientation[0]) < unhealthy_angle
        is_healthy_roll = np.abs(orientation[1]) < unhealthy_angle

        is_healthy = is_healthy_z and is_healthy_pitch and is_healthy_roll

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

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return (self.data.qpos[:, 2]) ** 2

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return np.sum(np.square(self.data.qvel[:2]))
        #return np.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        # Penalize non flat base orientation
        # TODO
        return 0
        #return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = self.data.qpos[2]
        init_height = self.init_qpos[2]
        return (base_height-self.init_height) ** 2

    def _reward_torques(self):
        # Penalize torques
        # TODO
        return 0
        #return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        # TODO
        return 0
        #return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        # TODO
        return 0
        #return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    def _reward_action_rate(self, before_joints, after_joints):
        # Penalize changes in actions
        return np.sum(np.square(before_joints - after_joints))
        #return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        # TODO
        return 0
        #return torch.sum(1. * (torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1),
        #                 dim=1)

    def _reward_termination(self):
        # Terminal reward / penalty
        # TODO
        pass
        #return self.reset_buf * ~self.time_out_buf

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        # TODO
        pass

        #out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.)  # lower limit
        #out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        #return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        # TODO
        pass

        #return torch.sum(
        #    (torch.abs(self.dof_vel) - self.dof_vel_limits * self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.),
        #    dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        # TODO
        pass

        #return torch.sum(
        #    (torch.abs(self.torques) - self.torque_limits * self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_tracking_lin_vel(self, before_pos, after_pos):
        # Tracking of linear velocity commands (xy axes)
        vel = after_pos-before_pos
        target_vel = np.array([0, 0, 1.0])
        vel_error = np.abs(target_vel-vel)

        return np.exp(-vel_error)

        #lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        #return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        # TODO
        pass
        #ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        #return torch.exp(-ang_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        # TODO
        pass
        # contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        # contact_filt = torch.logical_or(contact, self.last_contacts)
        # self.last_contacts = contact
        # first_contact = (self.feet_air_time > 0.) * contact_filt
        # self.feet_air_time += self.dt
        # rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact,
        #                         dim=1)  # reward only on first contact with the ground
        # rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1  # no reward for zero command
        # self.feet_air_time *= ~contact_filt
        # return rew_airTime

    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        # TODO
        pass
        #return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) > \
        #                 5 * torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)

    def _reward_stand_still(self):
        # Penalize motion at zero commands
        # TODO
        pass
        #return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (
        #            torch.norm(self.commands[:, :2], dim=1) < 0.1)

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        # TODO
        pass
        #return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :],
        #                             dim=-1) - self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)

    def step(self, delta_q):
        action = delta_q + self.data.qpos[-12:]
        action = np.clip(action, a_min=self.lower_limits, a_max=self.upper_limits)

        before_pos = self.data.qpos[:3].copy()
        before_vel = self.data.qvel[:3].copy()
        before_orientation = self.base_rotation
        before_joints = self.data.qpos[7:]
        self.do_simulation(action, self.frame_skip)
        after_pos = self.data.qpos[:3].copy()
        after_vel = self.data.qvel[:3].copy()
        after_orientation = self.base_rotation
        after_yaw = after_orientation[2]
        after_joints = self.data.qpos[7:]

        track_vel_reward = self._reward_tracking_lin_vel(before_vel, after_vel)
        healthy_reward = self._reward_healthy(scaling_factor=20.0)  # 1.0

        lin_vel_z_reward = self._reward_lin_vel_z()
        ang_vel_xy_reward = self._reward_ang_vel_xy()
        base_height_reward = self._reward_base_height()
        action_rate_reward = self._reward_action_rate()

        total_rewards = healthy_reward + track_vel_reward + lin_vel_z_reward + ang_vel_xy_reward + base_height_reward + action_rate_reward

        terminate = self.terminated
        observation = self._get_obs()
        info = {
            'total_reward': total_rewards,
            'traverse': self.data.qpos[0],
            'height': self.data.qpos[2]/self.init_joints[2],
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

    @staticmethod
    def print_rewards(info: dict, plot_all_rewards=True):

        for key in info.keys():
            print(key + str(info[key]))
        return
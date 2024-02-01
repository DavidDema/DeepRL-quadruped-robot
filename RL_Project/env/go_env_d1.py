from collections import defaultdict

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
                 cfg,
                 **kwargs,
                 ):
        self.env_cfg = cfg['env']
        self.rew_cfg = self.env_cfg['rewards']
        self.control_cfg = self.env_cfg['control']

        self._exclude_current_positions_from_observation = self.env_cfg['exclude_current_positions_from_observations']
        self._reset_noise_scale = self.env_cfg['reset_noise_scale']
        self.frame_skip = self.env_cfg['frame_skip']

        if self._exclude_current_positions_from_observation:
            self.obs_dim = 17 + 18
        else:
            self.obs_dim = 19 + 18

        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float64
        )

        self.use_torque = self.control_cfg['use_torque']
        if self.use_torque:
            scene = 'go/scene_torque.xml'
        else:
            scene = 'go/scene.xml'

        MujocoEnv.__init__(self,
                           model_path=os.path.join(os.path.dirname(__file__), scene),
                           frame_skip=self.frame_skip,
                           observation_space=observation_space,
                           **kwargs
                           )

        self.action_dim = 12
        self.action_space = Box(
            low=self.lower_limits, high=self.upper_limits, shape=(self.action_dim,), dtype=np.float64
        )

        self._healthy_z_range = self.env_cfg['healthy_z_range']
        self.unhealthy_angle = self.env_cfg['unhealthy_angle']
        self._terminate_when_unhealthy = self.env_cfg['terminate_when_unhealthy']

        self.only_positive_rewards = self.control_cfg['only_positive_rewards']
        self.p_gain = np.ones(self.action_dim) * self.control_cfg['stiffness']
        self.d_gain = np.ones(self.action_dim) * self.control_cfg['damping']
        self.torque_limits = self.control_cfg['torque_limits']

        self.vel_commands = self.control_cfg['vel_commands']
        self.base_height_target = self.control_cfg['base_height_target']

        self.base_pos = self.init_joints[:3]
        self.base_lin_vel = np.zeros(3)
        self.base_ang_vel = np.zeros(3)
        self.feet_pos = np.zeros(4)
        self.base_orientation = np.zeros(3)
        self.dof_pos = self.init_joints[-12:]
        self.dof_vel = np.zeros(12)
        self.dof_acc = np.zeros(12)
        self.action = np.zeros(12)
        self.torques = np.zeros(12)
        self.last_base_pos = self.init_joints[:3]
        self.last_base_lin_vel = np.zeros(3)
        self.last_feet_pos = np.zeros(3)
        self.last_base_orientation = np.zeros(3)
        self.last_dof_pos = self.init_joints[-12:]
        self.last_dof_vel = np.zeros(12)
        self.last_action = np.zeros(12)
        self.contact_forces = self.data.cfrc_ext

        self.timestep = 0
        self.max_timesteps = 1

    @property
    def lower_limits(self):
        # Limits der einzelnen Gelenke [oben rechts-links, oben vorne-hinten, unten vorne-hinten ]
        # return np.array([-0.863, -0.686, -2.818]*4) #(Anfangsdaten)
        return np.array([-0.25, -0.8, -1.318] * 4)

    @property
    def upper_limits(self):
        # return np.array([0.863, 4.501, -0.888]*4) #(Anfangsdaten)
        return np.array([0.25, 1.8, -0.888] * 4)

    @property
    def init_joints(self):
        # base position (x, y, z), base orientation (quaternion), 4x leg position (3 joints)
        return np.array([0, 0, 0.37, 1, 0, 0, 0] + [0, 0.7, -1.4] * 4)

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
        orientation = GOEnv.get_angle_degree(self.base_orientation)

        # Angles over "unhealthy_angle" degree are unhealthy
        is_healthy_pitch = np.abs(orientation[1]) < self.unhealthy_angle
        is_healthy_roll = np.abs(orientation[0]) < self.unhealthy_angle

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
    def _reward_unhealthy(self):
        # rewards unhealthy state (1=unhealthy)
        return -(self.is_healthy - 1)

    def _reward_not_living(self):
        # negative living reward
        k = 1.0
        return (k*(self.max_timesteps - self.timestep) / self.max_timesteps) ** 2

    def _reward_living(self):
        return (self.timestep / self.max_timesteps) ** 2

    def _reward_vel_x(self):
        k = -1.5
        #return np.exp(k*(self.vel_commands[0] - self.base_lin_vel[0]))
        return np.exp(-np.linalg.norm((self.vel_commands - self.base_lin_vel) ** 2))

    def _reward_vel_y(self):
        #return np.exp(-self.base_lin_vel[2] ** 2) # **2
        return self.base_lin_vel[1] ** 2

    def _reward_vel_z(self):
        return np.exp(-2.0 * self.base_lin_vel[2] ** 2)
        #return self.base_lin_vel[2] ** 2

    def _reward_x(self):
        # reward for a higher positive traverse (going further)
        if self.base_pos[0] > 0:
            # going front
            return self.base_pos[0]**2
        else:
            # going backwards
            return -2*self.base_pos[0]

    def _reward_y(self):
        # reward for a higher positive traverse (going further)
        return np.abs(self.base_pos[1]) ** 2

    def _reward_z(self):
        # penalize movement in z direction
        return np.exp(-2.0*np.abs(self.base_pos[2] - self.base_height_target) ** 2) # **2
        #return np.abs((self.base_pos[2] - self.base_height_target)) ** 2
        #return np.abs((self.base_pos[2] - self.base_height_target)) ** 2

    def _reward_yaw(self):
        return np.abs(self.base_orientation[2])
        #return np.abs(self.base_orientation[2]) ** 2

    def _reward_pitch(self):
        return np.abs(self.base_orientation[1])
        #return np.abs(self.base_orientation[1]) ** 2

    def _reward_roll(self):
        return np.abs(self.base_orientation[0])
        #return np.abs(self.base_orientation[0]) ** 2

    def _reward_yaw_rate(self):
        # penalty for high yaw rate
        return np.abs(self.base_ang_vel[2])
        #return np.abs(self.base_ang_vel[2]) ** 2

    def _reward_pitch_rate(self):
        # penalty for high pitch rate
        return np.abs(self.base_ang_vel[1])
        #return np.abs(self.base_ang_vel[1]) ** 2

    def _reward_roll_rate(self):
        # penalty for high roll rate
        return np.abs(self.base_ang_vel[0])
        #return np.abs(self.base_ang_vel[0]) ** 2


    def _reward_foot_slip(self, feet_pos, feet_vel, feet_cont, scaling_factor=1.0):  ##(YAN)## sehr viel kleiner machen
        if feet_cont.any():
            sum_velocity_squared = 0.0
            velocity_squared = 0.0

            for contact_index in feet_cont:
                # if 0 <= contact_index < len(feet_pos):
                velocity_squared = np.sum(feet_vel[contact_index][:2] ** 2)
                sum_velocity_squared += velocity_squared

                # print(velocity_squared)
            reward_slip = -scaling_factor * velocity_squared

        else:
            reward_slip = 0.0
        # print(reward_slip)
        return np.exp(reward_slip)
        ## scaling_factor * sum (boolean * feet_velocity_x,y)

    ##(YAN)## just optional, no positive reward, and penalty smaller

    def _reward_two_feet(self, feet_cont, scaling_factor = 1.0):
        #if len(feet_cont) < 2 or any(np.array_equal(feet_cont, pair) for pair in [[10, 28], [19, 37]]): ## optional, kleiner machen
        if len(feet_cont) < 1:
            reward_two_feet = -scaling_factor * 100
        else:
            reward_two_feet = scaling_factor * 10
        return reward_two_feet

    def _reward_feet_air_time(self):
        # Reward long steps
        # TODO
        return 0.0

    # ----------------------------------------------------- #
    def _reward_torques(self):
        # Penalize torques
        return np.sum(np.square(self.torques))

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return np.sum(np.square(self.dof_acc))

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return np.sum(np.square(self.dof_vel))

    def _reward_dof_pos(self):
        # reward joint positions similar to init position
        return np.sum(np.square(self.init_joints[7:] - self.dof_pos))

    def _reward_action_rate(self):
        # Penalize changes in actions
        return np.sum(np.square(self.last_action - self.action))

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return np.sum(1. * (np.linalg.norm(self.contact_forces) > 0.1))

    # ----------------------------------------------------- #
    # ETH
    # def _reward_tracking_lin_vel(self):
    #     # Tracking of linear velocity commands (xy axes)
    #     lin_vel_error = np.linalg.norm(self.vel_commands[:2] - self.base_lin_vel[:2])
    #     return np.exp(-lin_vel_error / self.tracking_sigma)
    #
    # def _reward_tracking_ang_vel(self):
    #     # Tracking of angular velocity commands (yaw)
    #     ang_vel_error = np.linalg.norm(self.vel_commands[2] - self.base_ang_vel[2])
    #     return np.exp(-ang_vel_error / self.tracking_sigma)

    def step(self, delta_q):

        self.last_base_pos = self.data.qpos[:3].copy()
        self.last_base_lin_vel = self.data.qvel[:3].copy()
        self.last_feet_pos = self.data.geom_xpos.copy()
        self.last_base_orientation = self.base_rotation
        self.last_dof_pos = self.data.qpos[-12:].copy()
        self.last_dof_vel = self.data.qvel[-12:].copy()

        #self.qt2 = self.qt1
        #self.qt1 = self.qt
        #self.qt = delta_q + self.init_joints[-12:]

        if self.use_torque:
            torque = (self.p_gain * (delta_q + self.init_joints[-12:] - self.last_dof_pos) -
                      self.d_gain * self.last_dof_vel)
            torque = np.clip(torque, a_min=-self.torque_limits, a_max=self.torque_limits)
        else:
            action = delta_q + self.last_dof_pos
            action = np.clip(action, a_min=self.lower_limits, a_max=self.upper_limits)
            torque = action
        self.do_simulation(torque, self.frame_skip)

        self.base_pos = self.data.qpos[:3].copy()
        self.base_lin_vel = self.data.qvel[:3].copy()

        self.base_orientation = self.base_rotation
        self.dof_pos = self.data.qpos[-12:].copy()
        self.dof_vel = self.data.qvel[-12:].copy()
        self.torques = self.data.qfrc_applied

        self.dof_acc = (self.last_dof_vel - self.dof_vel) / self.dt
        self.base_ang_vel = (self.last_base_orientation - self.base_orientation) / self.dt
        self.contact_forces = self.data.cfrc_ext

        self.feet_pos = self.data.geom_xpos.copy()
        #feet_vel = feet_pos / self.dt
        #feet_cont = self.data.contact.geom2

        rewards_dict = defaultdict(float)
        rewards_dict['unhealthy'] = self._reward_unhealthy()
        rewards_dict['not_living'] = self._reward_not_living()
        rewards_dict['living'] = self._reward_living()

        rewards_dict['vel_x'] = self._reward_vel_x()
        rewards_dict['vel_y'] = self._reward_vel_y()
        rewards_dict['vel_z'] = self._reward_vel_z()
        rewards_dict['x'] = self._reward_x()
        rewards_dict['y'] = self._reward_y()
        rewards_dict['z'] = self._reward_z()

        rewards_dict['yaw'] = self._reward_yaw()
        rewards_dict['pitch'] = self._reward_pitch()
        rewards_dict['roll'] = self._reward_roll()
        rewards_dict['yaw_rate'] = self._reward_yaw_rate()
        rewards_dict['pitch_rate'] = self._reward_pitch_rate()
        rewards_dict['roll_rate'] = self._reward_roll_rate()

        rewards_dict['foot_slip'] = 0 #self._reward_foot_slip(feet_pos, feet_vel, feet_cont)
        rewards_dict['two_feet'] = 0 #self._reward_two_feet()  # TODO
        rewards_dict['feet_air_time'] = 0 #self._reward_feet_air_time()

        # ETH
        rewards_dict['torques'] = self._reward_torques()
        rewards_dict['dof_acc'] = self._reward_dof_acc()
        rewards_dict['dof_vel'] = self._reward_dof_vel()
        rewards_dict['dof_pos'] = self._reward_dof_pos()
        rewards_dict['action_rate'] = self._reward_action_rate()
        rewards_dict['collision'] = self._reward_collision()

        # add reward scaling
        for key in rewards_dict.keys():
            rewards_dict[key] *= self.rew_cfg[key]


        total_rewards = sum(rewards_dict.values())
        if self.only_positive_rewards and total_rewards < 0:
            total_rewards = np.float64(0)

        terminate = self.terminated
        observation = self._get_obs()
        info = {
            'total_reward': total_rewards,
            'traverse': self.data.qpos[0],
            'side': self.data.qpos[1],
            'height': self.data.qpos[2],
            'roll': GOEnv.get_angle_degree(self.base_orientation[0]),
            'pitch': GOEnv.get_angle_degree(self.base_orientation[1]),
            'yaw': GOEnv.get_angle_degree(self.base_orientation[2]) / np.pi,
            'rewards': rewards_dict,
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
    def get_angle_degree(angle_rad):
        return 180*angle_rad/np.pi

    @staticmethod
    def print_rewards(info: dict):
        for key in info.keys():
            print(f"{key} : {info[key]:.2f}")
        return

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        # TODO
        raise NotImplementedError
        #pd controller
        actions_scaled = actions * self.action_scale
        control_type = self.control_type
        if control_type == "P":
            torques = self.p_gain*(actions_scaled + self.init_joints[-12:] - self.dof_pos) - self.d_gain*self.dof_vel
        elif control_type == "V":
            torques = self.p_gain*(actions_scaled - self.dof_vel) - self.d_gain*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type == "T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return np.clip(torques, -self.torque_limits, self.torque_limits)
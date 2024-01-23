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

        self._reset_noise_scale = self.env_cfg['reset_noise_scale']
        self._healthy_z_range = self.env_cfg['healthy_z_range']
        self._terminate_when_unhealthy = self.env_cfg['terminate_when_unhealthy']
        self._exclude_current_positions_from_observation = self.env_cfg['exclude_current_positions_from_observations']

        self.unhealthy_angle = self.env_cfg['unhealthy_angle']
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
        # TODO
        self.timestep = 0
        self.max_timesteps = 1

        self.only_positive_rewards = self.control_cfg['only_positive_rewards'] # TODO
        self.tracking_sigma = self.control_cfg['tracking_sigma']

        #self.use_torque = self.env_cfg['use_torque']

        self.control_type = self.control_cfg['control_type']
        self.vel_commands = self.control_cfg['vel_commands']
        self.base_height_target = self.control_cfg['base_height_target']

        self.base_pos = self.init_joints[:3]
        self.base_lin_vel = np.zeros(3)
        self.base_ang_vel = np.zeros(3)
        self.feet_pos = None
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

        self.p_gain = np.ones(self.action_dim) * self.control_cfg['stiffness']
        self.d_gain = np.ones(self.action_dim) * self.control_cfg['damping']

        self.torque_limits = 10.0

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
        orientation = 180 * np.array(self.base_rotation) / np.pi

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

        # print(self.data.geom_xpos[[10,19,28,37]])

        return np.concatenate([qpos, qvel])

    # ------------ reward functions----------------
    def _reward_healthy(self, scaling_factor=1.0):
        return scaling_factor * (self.is_healthy - 1)

    def _reward_living(self):
        k = 1.5
        return (k*(self.max_timesteps - self.timestep) / self.max_timesteps) ** 2

    def _reward_living_pos(self):
        return (self.timestep / self.max_timesteps) ** 2

    def _reward_lin_vel(self):
        return np.exp(-np.linalg.norm((self.vel_commands - self.base_lin_vel) ** 2))

    def _reward_z_vel(self):
        return np.exp(-self.base_lin_vel[2] ** 2) # **2

    def _reward_z_pos(self):
        # penalize movement in z direction
        return np.exp(-np.abs(self.base_pos[2] - self.base_height_target) ** 2) # **2

    def _reward_yaw(self):
        return np.abs(self.base_orientation[2]) ** 2

    def _reward_pitch(self):
        return np.abs(self.base_orientation[1]) ** 2

    def _reward_roll(self):
        return np.abs(self.base_orientation[0]) ** 2

    def _reward_yaw_rate(self):
        # penalty for high yaw rate
        return np.abs(self.base_ang_vel[2])

    def _reward_pitch_rate(self):
        # penalty for high pitch rate
        return np.abs(self.base_ang_vel[1])

    def _reward_roll_rate(self):
        # penalty for high roll rate
        return np.abs(self.base_ang_vel[0])

    def _reward_joint_pose(self):
        # reward joint positions similar to init position
        return np.linalg.norm(self.init_joints[7:] - self.dof_pos) ** 2

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
    '''
    def _reward_two_feet(self, feet_cont, scaling_factor = 1.0):
        #if len(feet_cont) < 2 or any(np.array_equal(feet_cont, pair) for pair in [[10, 28], [19, 37]]): ## optional, kleiner machen
        if len(feet_cont) < 1:    
            reward_two_feet = -scaling_factor * 100
        else:
            reward_two_feet = scaling_factor * 10
        return reward_two_feet
    '''

    # ----------------------------------------------------- #
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return np.square(self.base_lin_vel[2])

    def _reward_torques(self):
        # Penalize torques
        return np.sum(np.square(self.torques))

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return np.sum(np.square(self.dof_acc))

    def _reward_action_rate(self):
        # Penalize changes in actions
        return np.sum(np.square(self.last_action - self.action))

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = np.linalg.norm(self.vel_commands[:2] - self.base_lin_vel[:2])
        return np.exp(-lin_vel_error / self.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = np.linalg.norm(self.vel_commands[2] - self.base_ang_vel[2])
        return np.exp(-ang_vel_error / self.tracking_sigma)

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return np.sum(1. * (np.linalg.norm(self.contact_forces) > 0.1))

    def _reward_orientation(self):
        # Penalize non flat base orientation
        return np.sum(np.square(self.base_orientation[:2]))

    def _reward_base_height(self):
        # Penalize base height away from target
        return np.square(self.base_pos[2] - self.base_height_target)

    def _reward_going_far_x(self):
        reward_x = self.data.qpos[0]
        return reward_x

    def _reward_going_far_y(self):
        reward_y = self.data.qpos[1]
        return -np.abs(reward_y)

    def _reward_ang_vel_xy(self):
        # Penalize non flat base orientation
        return np.square(self.base_ang_vel[2])

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return np.sum(np.square(self.dof_vel))

    def _reward_feet_air_time(self):
        # Reward long steps
        return 0.0

    # ----------------------------------------------------- #

    def step(self, delta_q):

        self.last_base_pos = self.data.qpos[:3].copy()
        self.last_base_lin_vel = self.data.qvel[:3].copy()
        self.last_feet_pos = self.data.geom_xpos.copy()
        self.last_base_orientation = self.base_rotation
        self.last_dof_pos = self.data.qpos[-12:].copy()
        self.last_dof_vel = self.data.qvel[-12:].copy()

        action = delta_q + self.data.qpos[-12:]
        self.action = np.clip(action, a_min=self.lower_limits, a_max=self.upper_limits)

        if self.use_torque:
            torque = self.p_gain * delta_q - self.d_gain * self.data.qvel[-12:]
            torque = np.clip(torque, a_min=self.lower_limits, a_max=self.upper_limits)
        else:
            torque = self.action
        self.do_simulation(torque, self.frame_skip)

        self.base_pos = self.data.qpos[:3].copy()
        self.base_lin_vel = self.data.qvel[:3].copy()
        self.feet_pos = self.data.geom_xpos.copy()
        self.base_orientation = self.base_rotation
        self.dof_pos = self.data.qpos[-12:].copy()
        self.dof_vel = self.data.qvel[-12:].copy()
        self.torques = self.data.qfrc_applied  # self._compute_torques(self.action) # which torque ?

        self.dof_acc = (self.last_dof_vel - self.dof_vel) / self.dt
        self.base_ang_vel = (self.last_base_orientation - self.base_orientation) / self.dt
        self.contact_forces = self.data.cfrc_ext

        feet_pos = self.data.geom_xpos  # [[10,19,28,37]]
        feet_vel = feet_pos / self.dt
        feet_cont = self.data.contact.geom2

        track_vel_reward = self._reward_lin_vel()
        living_reward = self._reward_living()
        living_pos_reward = self._reward_living_pos()
        healthy_reward = self._reward_healthy()

        yaw_rate_reward = self._reward_yaw_rate()
        pitch_rate_reward = self._reward_pitch_rate()
        roll_rate_reward = self._reward_roll_rate()
        yaw_reward = self._reward_yaw()
        pitch_reward = self._reward_pitch()
        roll_reward = self._reward_roll()

        joint_pos_reward = self._reward_joint_pose()
        z_vel_reward = self._reward_z_vel()
        z_pos_reward = self._reward_z_pos()
        foot_slip_reward = self._reward_foot_slip(feet_pos, feet_vel, feet_cont)
        two_feet_reward = 0 # TODO

        # ETH
        # tracking_lin_vel_reward = self._reward_tracking_lin_vel()
        # tracking_ang_vel_reward = self._reward_tracking_ang_vel()
        # lin_vel_z_reward = self._reward_base_height()
        # ang_vel_xy_reward = self._reward_ang_vel_xy()
        # torques = self._reward_torques()
        # dof_acc = self._reward_dof_acc()
        # feet_air_time = self._reward_feet_air_time()
        # collision = self._reward_collision()
        # action_rate = self._reward_action_rate()

        rewards = [
            healthy_reward * self.rew_cfg['healthy'],       # 5.0
            track_vel_reward * self.rew_cfg['lin_vel'],     # 15.0
            living_reward * self.rew_cfg['living'],         # -2.0
            living_pos_reward * self.rew_cfg['living_pos'], # 2.0
            yaw_rate_reward * self.rew_cfg['yaw_rate'],     # -2.0
            pitch_rate_reward * self.rew_cfg['pitch_rate'], # -2.0
            roll_rate_reward * self.rew_cfg['roll_rate'],   # -2.0
            yaw_reward * self.rew_cfg['yaw'],               # -0.3
            pitch_reward * self.rew_cfg['pitch'],           # -0.5
            roll_reward * self.rew_cfg['roll'],             # -0.5
            joint_pos_reward * self.rew_cfg['joint_pos'],   # -0.3
            z_vel_reward * self.rew_cfg['z_vel'],           # 2.0
            z_pos_reward * self.rew_cfg['z_pos'],           # 2.0
            foot_slip_reward * self.rew_cfg['foot_slip'],   # 0.3
            two_feet_reward * self.rew_cfg['two_feet'],     # 0.0
            # ETH
            # tracking_lin_vel_reward * self.rew_cfg['tracking_lin_vel'],
            # tracking_ang_vel_reward * self.rew_cfg['tracking_ang_vel'],
            # lin_vel_z_reward * self.rew_cfg['lin_vel_z'],
            # ang_vel_xy_reward * self.rew_cfg['ang_vel_xy'],
            # torques * self.rew_cfg['torques'],
            # dof_acc * self.rew_cfg['dof_acc'],
            # feet_air_time * self.rew_cfg['feet_air_time'],
            # collision * self.rew_cfg['collision'],
            # action_rate * self.rew_cfg['action_rate'],
        ]

        total_rewards = np.sum(rewards)

        terminate = self.terminated
        observation = self._get_obs()
        info = {
            'total_reward': total_rewards,
            'joint_pos_reward': joint_pos_reward,
            # TODO
            'yaw_rate_reward': yaw_rate_reward,
            'track_vel_reward': track_vel_reward,
            'healthy_reward': healthy_reward,
            'living_reward': living_reward,
            'z_pos_reward': z_pos_reward,
            'z_vel_reward': z_vel_reward,
            'traverse': self.data.qpos[0],
            'side': self.data.qpos[1],
            'height': self.data.qpos[2],
            'roll': 180*self.base_orientation[0]/np.pi,
            'pitch': 180*self.base_orientation[1]/np.pi,
            'yaw': 180 * self.base_orientation[2] / np.pi,
            'feet_slip': foot_slip_reward,
            'two feet': two_feet_reward,
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
    def print_rewards(info: dict):
        #info.pop('traverse')
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
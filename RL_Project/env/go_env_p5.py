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
        # Limits der einzelnen Gelenke [oben rechts-links, oben vorne-hinten, unten vorne-hinten ]
        # return np.array([-0.863, -0.686, -2.818]*4) #(Anfangsdaten)
        # return np.array([-0.00, -0.5, -1.00]*4)
        return np.array([-0.25, -0.8, -1.318] * 4)

    @property
    def upper_limits(self):
        # return np.array([0.863, 4.501, -0.888]*4) #(Anfangsdaten)
        # return np.array([0.00, 1.0, 0.00]*4)
        return np.array([0.25, 1.8, -0.888] * 4)

    @property
    def init_joints(self):
        # base position (x, y, z), base orientation (quaternion), 4x leg position (3 joints)
        # return np.array([0, 0, 0.37, 1, 0, 0, 0] + [0, 0.7, -1.4]*4)
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
        orientation = 360 * np.array(self.base_rotation) / (2 * np.pi)

        # Angles over "unhealthy_angle" degree are unhealthy
        unhealthy_angle = 40
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

        # print(self.data.geom_xpos[[10,19,28,37]])

        return np.concatenate([qpos, qvel])

    # ------------ reward functions----------------
    def _reward_healthy(self, scaling_factor=1.0):
        return scaling_factor * (self.is_healthy - 1)

    def _reward_living(self, timestep=0, max_timesteps=1, scaling_factor=2.0, positive=True):
        """Normalized living reward.
        '-1' at begin and '+0' if lived long time (until the end of episode)"""
        if positive:
            #return scaling_factor * (1 - ((max_timesteps - timestep) / max_timesteps) ** 2)
            return scaling_factor * np.exp(-2.0 * ((max_timesteps - timestep) / max_timesteps))
        else:
            return -scaling_factor * (((max_timesteps - timestep) / max_timesteps) ** 2)

    def _reward_lin_vel(self, before_pos, after_pos, scaling_factor=10.0):
        # target_vel = np.array([0.5, 0, 0]) (Ausgangswerte)
        target_vel = np.array([0.3, 0, 0])
        lin_vel = (after_pos - before_pos) / self.dt
        return scaling_factor * np.exp(
            -np.linalg.norm((target_vel - lin_vel) ** 2))  ## **2 von mir hinzugefügt (Piet), nature Paper;

    def _reward_z_vel(self, before_pos, after_pos,
                      scaling_factor=1.0):  ##(YAN)## should be very small, in total not so important, velocity is a big value
        # penalize movement in z direction
        z_vel = np.abs((after_pos[2] - before_pos[2])) / self.dt
        # return -scaling_factor * z_vel**2
        return -scaling_factor * z_vel ** 2

    def _reward_z_pos(self, after_pos, scaling_factor=10.0):
        # penalize movement in z direction
        # dz = after_pos[2]-self.init_joints[2] ##(YAN)## init.joints lower to 0.5* oder kleiner!
        dz = after_pos[2] - 0.5 * self.init_joints[2]
        # return -scaling_factor * dz**2
        return -scaling_factor * dz ** 2

    def _reward_pitch_roll(self, orientation, scaling_factor=10.0):
        # penalty for non-flat base orientation
        pitch_penalty = np.abs(orientation[0])  # pitch
        roll_penalty = np.abs(orientation[1])  # roll
        return -scaling_factor * (pitch_penalty ** 2 + roll_penalty ** 2)

    def _reward_yaw(self, yaw, before_pos, after_pos, scaling_factor=10.0):
        # reward movement in forward direction
        direction = (after_pos[:2] - before_pos[:2])
        target_yaw = np.arctan2(direction[1], direction[0])
        return -scaling_factor * np.linalg.norm(yaw - target_yaw)

    def _reward_yaw_rate(self, before_orientation, after_orientation, scaling_factor=10.0):
        # penalty for high yaw rate
        acceleration = np.abs((after_orientation[2] - before_orientation[2]) / self.dt)
        return -scaling_factor * acceleration

    def _reward_pitch_roll_rate(self, before_orientation, after_orientation, scaling_factor=10.0):
        # penalty for high pitch and roll rate
        acceleration = np.abs((after_orientation[:2] - before_orientation[:2]) / self.dt)
        return -scaling_factor * acceleration.mean()

    def _reward_joint_pose(self, current_joints, init_joints, scaling_factor=10.0):
        # reward joint positions similar to init position
        return -scaling_factor * np.linalg.norm(init_joints - current_joints) ** 2  ## auch hier **2 Piet hinzugefügt

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

    def step(self, delta_q):
        a = 2.0 * np.ones((12,))
        b = 0.5 * np.ones((12,))
        torque = a * delta_q - b * self.data.qvel[-12:]
        action = torque

        #action = delta_q + self.data.qpos[-12:]  ###(YAN)### hier auf jedenfall noch den Torque berechnen
        action = np.clip(action, a_min=self.lower_limits, a_max=self.upper_limits)

        before_pos = self.data.qpos[:3].copy()
        before_vel = self.data.qvel[:3].copy()
        before_orientation = self.base_rotation
        self.do_simulation(action, self.frame_skip)
        after_pos = self.data.qpos[:3].copy()
        after_vel = self.data.qvel[:3].copy()
        after_orientation = self.base_rotation
        after_yaw = after_orientation[2]
        after_joints = self.data.qpos[7:]

        # -------------------
        # -------------------
        # Rewards
        track_vel_reward = self._reward_lin_vel(before_pos, after_pos,
                                                scaling_factor=15.0)  # 1.5; fkt 0.5 ##(YAN)## should be the biggest

        living_reward = self._reward_living(timestep=self.timestep, max_timesteps=self.max_timesteps,
                                            scaling_factor=2.0, positive=False)  # 3.5; fkt 5

        healthy_reward = self._reward_healthy(
            scaling_factor=5.0)  # 1.0; fkt 15 ##(YAN)##increase, damit er nicht einfach vorne über fällt!
        joint_pos_reward = self._reward_joint_pose(after_joints, self.init_joints[7:],
                                                   scaling_factor=0.3)  # 0.3; fkt 0.3
        z_vel_reward = self._reward_z_vel(before_pos=before_pos, after_pos=after_pos,
                                          scaling_factor=2.0)  # 2.0; fkt 2.0
        z_pos_reward = self._reward_z_pos(after_pos=after_pos, scaling_factor=2.0)  # 2.0; fkt 2.0

        # -------------------
        # Orientation Rewards
        yaw_rate_reward = self._reward_yaw_rate(before_orientation, after_orientation,
                                                scaling_factor=0.5)  # 0.8; fkt 0.2
        pitchroll_rate_reward = self._reward_pitch_roll_rate(before_orientation, after_orientation,
                                                             scaling_factor=2.0)  # 0.05; fkt 0.2
        pitchroll_reward = self._reward_pitch_roll(after_orientation, scaling_factor=2.0)  # 5.0; fkt 2.0
        orient_reward = self._reward_yaw(after_yaw, before_pos, after_pos, scaling_factor=0.3)  # 0.1; fkt 0.1

        # -------------------
        # Feet Rewards
        feet_pos = self.data.geom_xpos  # [[10,19,28,37]]
        feet_vel = feet_pos / self.dt
        feet_cont = self.data.contact.geom2

        foot_slip_reward = self._reward_foot_slip(feet_pos, feet_vel, feet_cont, scaling_factor=0.3)  # fkt 5.0
        # two_feet_reward = self._reward_two_feet(feet_cont, scaling_factor = 2.0) # fkt 5.0 ##(YAN)## should work without, just optional



        total_rewards = track_vel_reward + living_reward + (yaw_rate_reward + pitchroll_reward + \
                                                            orient_reward + pitchroll_rate_reward + z_vel_reward + z_pos_reward + \
                                                            foot_slip_reward)

        terminate = self.terminated
        observation = self._get_obs()
        info = {
            'total_reward': total_rewards,
            'joint_pos_reward': joint_pos_reward,
            'pitchroll_rate_reward': pitchroll_rate_reward,
            'orient_reward': orient_reward,
            'pitchroll_reward': pitchroll_reward,
            'yaw_rate_reward': yaw_rate_reward,
            'track_vel_reward': track_vel_reward,
            'healthy_reward': healthy_reward,
            'living_reward': living_reward,
            'z_pos_reward': z_pos_reward,
            'z_vel_reward': z_vel_reward,
            'traverse': self.data.qpos[0],
            'height': self.data.qpos[2],
            'roll': after_orientation[1],
            'feet_slip': foot_slip_reward,
            # 'two feet': two_feet_reward
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

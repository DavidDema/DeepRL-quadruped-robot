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

    class Cfg:
        class RewardScale:
            lin_vel = 15.0
            living = -2.0 
            healthy = 5.0
            yaw_rate = -2.0
            pitchroll_rate = -2.0
            pitchroll = -0.5
            joint_pos = -0.0 # -10.0
            orientation = -0.3
            z_vel = 1.0 # 2.0
            z_pos = 1.0 # 2.0
            foot_slip = 1.0 # 0.3 

            tracking_lin_vel = 0.0 # 1.0
            tracking_ang_vel = 0.0 # 0.5
            lin_vel_z = 0.0 # -2.0
            ang_vel_xy = 0.0 # -0.05
            torques = 0.0 # -0.00001
            dof_acc = 0.0 # -2.5e-7
            feet_air_time = 0.0 #  1.0
            collision = 0.0 # -1.0
            action_rate = 0.0 # -0.01
            
            termination = -0.0
            dof_vel = -0.
            base_height = -0. 
            feet_stumble = -0.0 
            stand_still = -0.

        class Control:
            stiffness = 2.0 # 15 
            damping = 0.5 # 1.5
            action_scale = 1.0
        
        def __init__(self):
            self.base_height_target = 0.34
            self.reward_scale = self.RewardScale()
            self.control = self.Control()
            self.commands = [0.5, 0.0, 0.0] # x, y, yaw
            self.tracking_sigma = 0.1
            self.only_positive_rewards = False


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

        self.cfg = self.Cfg()

        self.action = np.zeros(12)
        self.last_action = np.zeros(12)

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

        self.p_gain = np.ones(self.action_dim) * self.cfg.control.stiffness
        self.d_gain = np.ones(self.action_dim) * self.cfg.control.damping


    @property
    def lower_limits(self):
        return np.array([-0.25, -0.8, -1.318] * 4)

    @property
    def upper_limits(self):
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
    def _reward_healthy(self):
        return (self.is_healthy - 1)

    def _reward_living(self, timestep=0, max_timesteps=1):
        return ((max_timesteps - timestep) / max_timesteps) ** 2

    def _reward_lin_vel(self):
        return np.exp(-np.linalg.norm((self.cfg.commands - self.base_lin_vel) ** 2))

    def _reward_z_vel(self):
        return np.exp(-self.base_lin_vel[2])
 
    def _reward_z_pos(self):
        # penalize movement in z direction
        return np.exp(-np.abs(self.base_pos[2] - self.cfg.base_height_target))

    def _reward_pitch_roll(self):
        # penalty for non-flat base orientation
        pitch_penalty = np.abs(self.base_orientation[0])  # pitch
        roll_penalty = np.abs(self.base_orientation[1])  # roll
        return pitch_penalty ** 2 + roll_penalty ** 2

    def _reward_yaw(self):
        # reward movement in forward direction
        direction = (self.base_pos[:2] - self.last_base_pos[:2])
        target_yaw = np.arctan2(direction[1], direction[0])
        return np.linalg.norm(self.base_orientation[2] - target_yaw)

    def _reward_yaw_rate(self):
        # penalty for high yaw rate
        return np.abs((self.base_orientation[2] - self.last_base_orientation[2]) / self.dt)

    def _reward_pitch_roll_rate(self):
        # penalty for high pitch and roll rate
        return np.sum(np.abs((self.base_orientation[:2] - self.last_base_orientation[:2]) / self.dt))

    def _reward_joint_pose(self):
        # reward joint positions similar to init position
        return np.linalg.norm(self.init_joints[7:] - self.dof_pos) ** 2 

    def _reward_foot_slip(self, feet_pos, feet_vel, feet_cont, scaling_factor=1.0):
        if feet_cont.any():
            sum_velocity_squared = 0.0
            velocity_squared = 0.0

            for contact_index in feet_cont:
                velocity_squared = np.sum(feet_vel[contact_index][:2] ** 2)
                sum_velocity_squared += velocity_squared

            reward_slip = -scaling_factor * velocity_squared
        else:
            reward_slip = 0.0

        return np.exp(reward_slip)
    
    # ----------------------------------------------------- #
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return np.square(self.base_lin_vel[2])
        
    def _reward_torques(self):
        # Penalize torques
        return np.sum(np.square(self.last_torques))
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return np.sum(np.square((self.dof_acc)))
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return np.sum(np.square(self.last_action - self.action))
    
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = np.linalg.norm(self.cfg.commands[:2] - self.base_lin_vel[:2])
        return np.exp(-lin_vel_error/self.cfg.tracking_sigma)
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = np.linalg.norm(self.cfg.commands[2] - self.base_ang_vel[2])
        return np.exp(-ang_vel_error/self.cfg.tracking_sigma)
    
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return np.sum(1.*(np.linalg.norm(self.contact_forces) > 0.1))

    def _reward_orientation(self):
        # Penalize non flat base orientation
        return np.sum(np.square(self.base_orientation[:2]))

    def _reward_base_height(self):
        # Penalize base height away from target
        return np.square(self.base_pos[2] - self.cfg.base_height_target)
    
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

        self.action = delta_q + self.data.qpos[-12:]
        self.last_torques = self.p_gain * delta_q - self.d_gain * self.data.qvel[-12:]
        #action = torque
        self.action = np.clip(self.action, a_min=self.lower_limits, a_max=self.upper_limits)

        self.do_simulation(self.action, self.frame_skip)

        self.base_pos = self.data.qpos[:3].copy()
        self.base_lin_vel = self.data.qvel[:3].copy()
        self.feet_pos = self.data.geom_xpos.copy()
        self.base_orientation = self.base_rotation
        self.dof_pos = self.data.qpos[-12:].copy()
        self.dof_vel = self.data.qvel[-12:].copy()
        self.action = delta_q + self.data.qpos[-12:].copy()
        self.action = np.clip(self.action, a_min=self.lower_limits, a_max=self.upper_limits)
        self.torques = self.data.qfrc_applied # self._compute_torques(self.action) # which torque ?
        
        self.dof_acc = (self.last_dof_vel - self.dof_vel) / self.dt  
        self.base_ang_vel = (self.last_base_orientation - self.base_orientation) / self.dt 
        self.contact_forces = self.data.cfrc_ext

        feet_pos = self.data.geom_xpos  # [[10,19,28,37]]
        feet_vel = feet_pos / self.dt
        feet_cont = self.data.contact.geom2

        track_vel_reward = self._reward_lin_vel() 
        living_reward = self._reward_living(timestep=self.timestep, max_timesteps=self.max_timesteps)
        healthy_reward = self._reward_healthy()
        yaw_rate_reward = self._reward_yaw_rate() 
        pitchroll_rate_reward = self._reward_pitch_roll_rate() 
        pitchroll_reward = self._reward_pitch_roll()
        joint_pos_reward = self._reward_joint_pose()
        orient_reward = self._reward_yaw()
        z_vel_reward = self._reward_z_vel()
        z_pos_reward = self._reward_z_pos()
        foot_slip_reward = self._reward_foot_slip(feet_pos, feet_vel, feet_cont, scaling_factor=0.3)
        
        # ETH
        tracking_lin_vel_reward = self._reward_tracking_lin_vel()
        tracking_ang_vel_reward = self._reward_tracking_ang_vel()
        lin_vel_z_reward = self._reward_base_height()
        ang_vel_xy_reward = self._reward_ang_vel_xy()
        torques = self._reward_torques()
        dof_acc = self._reward_dof_acc()
        feet_air_time = self._reward_feet_air_time()
        collision = self._reward_collision()
        action_rate = self._reward_action_rate()

        rewards = [
            healthy_reward        * self.cfg.reward_scale.healthy,
            track_vel_reward      * self.cfg.reward_scale.lin_vel, 
            living_reward         * self.cfg.reward_scale.living,
            yaw_rate_reward       * self.cfg.reward_scale.yaw_rate,
            pitchroll_reward      * self.cfg.reward_scale.pitchroll,
            orient_reward         * self.cfg.reward_scale.orientation,
            pitchroll_rate_reward * self.cfg.reward_scale.pitchroll_rate,
            z_vel_reward          * self.cfg.reward_scale.z_vel,
            z_pos_reward          * self.cfg.reward_scale.z_pos,
            foot_slip_reward      * self.cfg.reward_scale.foot_slip,
            # ETH
            tracking_lin_vel_reward * self.cfg.reward_scale.tracking_lin_vel,
            tracking_ang_vel_reward * self.cfg.reward_scale.tracking_ang_vel,
            lin_vel_z_reward * self.cfg.reward_scale.lin_vel_z,
            ang_vel_xy_reward * self.cfg.reward_scale.ang_vel_xy,
            torques * self.cfg.reward_scale.torques,
            dof_acc * self.cfg.reward_scale.dof_acc,
            feet_air_time * self.cfg.reward_scale.feet_air_time,
            collision * self.cfg.reward_scale.collision,
            action_rate * self.cfg.reward_scale.action_rate,
        ]

        total_rewards = np.sum(rewards)

        terminate = self.terminated
        observation = self._get_obs()
        info = {
            'total_reward': total_rewards,
            'tracking_lin_vel_reward': tracking_lin_vel_reward,
            'tracking_ang_vel_reward': tracking_ang_vel_reward,
            'lin_vel_z_reward': lin_vel_z_reward,
            'ang_vel_xy_reward': ang_vel_xy_reward,
            'torques': torques,
            'dof_acc': dof_acc,
            'feet_air_time': feet_air_time,
            'collision': collision,
            'action_rate': action_rate,
            # 'joint_pos_reward': joint_pos_reward,
            # 'pitchroll_rate_reward': pitchroll_rate_reward,
            # 'orient_reward': orient_reward,
            # 'pitchroll_reward': pitchroll_reward,
            # 'yaw_rate_reward': yaw_rate_reward,
            # 'track_vel_reward': track_vel_reward,
            # 'healthy_reward': healthy_reward,
            # 'living_reward': living_reward,
            # 'z_pos_reward': z_pos_reward,
            # 'z_vel_reward': z_vel_reward,
            # 'feet_slip': foot_slip_reward,
            'traverse': self.data.qpos[0],
            'height': self.data.qpos[2],
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

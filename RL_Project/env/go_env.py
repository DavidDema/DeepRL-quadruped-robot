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
        "render_fps": 12,
    }

    class Cfg:
        class RewardScale:
            lin_vel = 0.0
            living = 0.0 
            healthy = 0.0
            yaw_rate = 0.0
            pitchroll_rate = 0.0
            pitchroll = 0.0
            joint_pos = 0.0 
            orientation = 0.0
            z_vel = 0.0
            z_pos = 0.0 
            foot_slip = 0.0

            termination = -0.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            torques = -0.00001
            dof_vel = -0.
            dof_acc = -2.5e-7
            base_height = -0. 
            feet_air_time = 0.0 # 1.0 # not implemented!
            collision = -1.0
            feet_stumble = -0.0 
            action_rate = -0.01             
            stand_still = -0.

        class Control:
            control_type = 'P'
            stiffness = 10.0 # 15 
            damping = 1.0 # 1.5
            action_scale = 0.5
        
        def __init__(self):
            self.base_height_target = 0.34
            self.reward_scale = self.RewardScale()
            self.control = self.Control()
            self.commands = [0.5, 0.0, 0.0] # x, y, yaw
            self.tracking_sigma = 0.25
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
                           model_path=os.path.join(os.path.dirname(__file__), 'go/go1_unitree.xml'),
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
        self.last_feet_pos = None
        self.last_base_orientation = np.zeros(3)
        self.last_dof_pos = self.init_joints[-12:]
        self.last_dof_vel = np.zeros(12)
        self.last_action = np.zeros(12)
        self.contact_forces = self.data.cfrc_ext
        self.feet_contact = None
        self.feet_vel = None

        self.p_gain = np.ones(self.action_dim) * self.cfg.control.stiffness
        self.d_gain = np.ones(self.action_dim) * self.cfg.control.damping

        self.torque_limits = 10.0

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

    def _reward_foot_slip(self):
        reward = 0.0
        if self.feet_contact.any():
            for contact_index in self.feet_contact:
                reward += np.sum(self.feet_vel[contact_index][:2] ** 2)
        return reward
    
    # ----------------------------------------------------- #
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return np.square(self.base_lin_vel[2])
        
    def _reward_torques(self):
        # Penalize torques
        return np.sum(np.square(self.torques))
    
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
    
    def _reward_termination(self):
        # Terminal reward / penalty
        return self.terminated
    
    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return 0.0
        
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return 0.0

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return 0.0


    # ----------------------------------------------------- #

    def step(self, action):

        self.last_base_pos = self.data.qpos[:3].copy()
        self.last_base_lin_vel = self.data.qvel[:3].copy()
        self.last_feet_pos = self.data.geom_xpos.copy()
        self.last_base_orientation = self.base_rotation
        self.last_dof_pos = self.data.qpos[-12:].copy()
        self.last_dof_vel = self.data.qvel[-12:].copy()
        
        self.last_action = self.action
        self.action = np.clip(action, a_min=self.lower_limits, a_max=self.upper_limits)

        if True:
            torque = self._compute_torques(self.action)
        else:
            torque = self.action
        self.do_simulation(torque, self.frame_skip)

        self.base_pos = self.data.qpos[:3].copy()
        self.base_lin_vel = self.data.qvel[:3].copy()
        self.feet_pos = self.data.geom_xpos.copy() # [[10,19,28,37]]
        self.base_orientation = self.base_rotation
        self.dof_pos = self.data.qpos[-12:].copy()
        self.dof_vel = self.data.qvel[-12:].copy()
        self.torques = self.data.qfrc_applied
        
        self.dof_acc = (self.last_dof_vel - self.dof_vel) / self.dt  
        self.base_ang_vel = (self.last_base_orientation - self.base_orientation) / self.dt 
        self.contact_forces = self.data.cfrc_ext

        self.feet_vel = (self.last_feet_pos - self.feet_pos) / self.dt
        self.feet_contact = self.data.contact.geom2

        track_vel_reward = self._reward_lin_vel() * self.cfg.reward_scale.lin_vel 
        living_reward = self._reward_living(timestep=self.timestep, max_timesteps=self.max_timesteps) * self.cfg.reward_scale.living
        healthy_reward = self._reward_healthy() * self.cfg.reward_scale.healthy
        yaw_rate_reward = self._reward_yaw_rate() * self.cfg.reward_scale.yaw_rate 
        pitchroll_rate_reward = self._reward_pitch_roll_rate() * self.cfg.reward_scale.pitchroll_rate 
        pitchroll_reward = self._reward_pitch_roll() * self.cfg.reward_scale.pitchroll
        joint_pos_reward = self._reward_joint_pose() * self.cfg.reward_scale.joint_pos
        orient_reward = self._reward_yaw() * self.cfg.reward_scale.orientation
        z_vel_reward = self._reward_z_vel() * self.cfg.reward_scale.z_vel
        z_pos_reward = self._reward_z_pos() * self.cfg.reward_scale.z_pos
        foot_slip_reward = self._reward_foot_slip() * self.cfg.reward_scale.foot_slip
        
        # ETH
        tracking_lin_vel_reward = self._reward_tracking_lin_vel() * self.cfg.reward_scale.tracking_lin_vel
        tracking_ang_vel_reward = self._reward_tracking_ang_vel() * self.cfg.reward_scale.tracking_ang_vel
        lin_vel_z_reward = self._reward_lin_vel_z() * self.cfg.reward_scale.lin_vel_z
        ang_vel_xy_reward = self._reward_ang_vel_xy() * self.cfg.reward_scale.ang_vel_xy
        torques_reward = self._reward_torques() * self.cfg.reward_scale.torques
        dof_acc_reward = self._reward_dof_acc() * self.cfg.reward_scale.dof_acc
        feet_air_time_reward = self._reward_feet_air_time() * self.cfg.reward_scale.feet_air_time
        collision_reward = self._reward_collision() * self.cfg.reward_scale.collision
        action_rate_reward = self._reward_action_rate() * self.cfg.reward_scale.action_rate
        termination_reward = self._reward_termination() * self.cfg.reward_scale.termination
        dof_vel_reward = self._reward_dof_vel() * self.cfg.reward_scale.dof_vel
        base_height_reward = self._reward_base_height() * self.cfg.reward_scale.base_height
        feet_stumble_reward = self._reward_stumble() * self.cfg.reward_scale.feet_stumble
        stand_still_reward = self._reward_stand_still() * self.cfg.reward_scale.stand_still

        rewards = [
            healthy_reward,
            track_vel_reward, 
            living_reward,
            yaw_rate_reward,
            pitchroll_reward,
            orient_reward,
            joint_pos_reward,
            pitchroll_rate_reward,
            z_vel_reward,
            z_pos_reward,
            foot_slip_reward,
            # ETH
            tracking_lin_vel_reward,
            tracking_ang_vel_reward,
            lin_vel_z_reward,
            ang_vel_xy_reward,
            torques_reward,
            dof_acc_reward,
            feet_air_time_reward,
            collision_reward,
            action_rate_reward,            
            termination_reward,
            dof_vel_reward,
            base_height_reward,
            feet_stumble_reward,
            stand_still_reward,
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
            'torques_reward': torques_reward,
            'dof_acc_reward': dof_acc_reward,
            'feet_air_time_reward': feet_air_time_reward,
            'collision_reward': collision_reward,
            'action_rate_reward': action_rate_reward,
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
            'feet_slip': foot_slip_reward,
            'termination_reward': termination_reward,
            'dof_vel_reward': dof_vel_reward,
            'base_height_reward': base_height_reward,
            'feet_stumble_reward': feet_stumble_reward,
            'stand_still_reward': stand_still_reward,
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

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type
        if control_type=="P":
            torques = self.p_gain*(actions_scaled + self.init_joints[-12:] - self.dof_pos) - self.d_gain*self.dof_vel
        elif control_type=="V":
            torques = self.p_gain*(actions_scaled - self.dof_vel) - self.d_gain*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return np.clip(torques, -self.torque_limits, self.torque_limits)
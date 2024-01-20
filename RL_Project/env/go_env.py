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
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            torques = -0.00001
            dof_acc = -2.5e-7
            feet_air_time =  1.0
            collision = -1.
            action_rate = -0.01
            # --- apparently not used --- #
            termination = 0.0
            orientation = 0.0
            dof_vel = 0.0
            base_height = 0.0
            feet_stumble = 0.0
            stand_still = 0.0

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
        self.action = np.zeros(12)
        self.torques = np.zeros(12)
        self.last_base_pos = self.init_joints[:3]
        self.last_base_vel = np.zeros(3)
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
        return np.array([-0.1, 0.2, -1.6]*4)

    @property
    def upper_limits(self):
        return np.array([0.1, 0.7, -1.9]*4)

    @property
    def init_joints(self):
        # base position (x, y, z), base orientation (quaternion), 4x leg position (3 joints) 
        return np.array([0, 0, 0.34, 1, 0, 0, 0] + [0, 0.6, -1.8]*4)

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
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return np.square(self.base_lin_vel[2])

    def _reward_ang_vel_xy(self):
        # Penalize non flat base orientation
        return np.sum(np.square(self.base_ang_vel[2]))
    
    def _reward_torques(self):
        # Penalize torques
        return np.sum(np.square(self.torques))
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return np.sum(np.square((self.last_dof_vel - self.dof_vel) / self.dt))
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return np.sum(np.square(self.last_action - self.action))

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = np.sum(np.square(self.cfg.commands[:2] - self.base_lin_vel[:2]))
        return np.exp(-lin_vel_error/self.cfg.tracking_sigma)
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = np.square(self.cfg.commands[2] - self.base_ang_vel[2])
        return np.exp(-ang_vel_error/self.cfg.tracking_sigma)
    
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return np.sum(1.*(np.linalg.norm(self.contact_forces) > 0.1))

    def _reward_feet_air_time(self):
        # Reward long steps
        return 0.0
    
    # --- BEGIN: apparently not used reward functions --- # 

    def _reward_orientation(self):
        # Penalize non flat base orientation
        return np.sum(np.square(self.base_orientation[:2]))

    def _reward_base_height(self):
        # Penalize base height away from target
        return np.square(self.base_pos[2] - self.cfg.base_height_target)
    
    def _reward_dof_vel(self):
        # Penalize dof velocities
        return np.sum(np.square(self.dof_vel))
    
    def _reward_termination(self):
        # Terminal reward / penalty
        return 0.0
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        return 0.0

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        return 0.0

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return 0.0
    
    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return 0.0
        
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return 0.0

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return 0.0
    
    # --- END: apparently not used reward functions --- # 

    def step(self, delta_q):
        
        self.base_pos = self.data.qpos[:3].copy()
        self.base_lin_vel = self.data.qvel[:3].copy()
        self.feet_pos = self.data.geom_xpos.copy()
        self.base_orientation = self.base_rotation
        self.dof_pos = self.data.qpos[7:]
        self.dof_vel = (self.last_dof_pos - self.dof_pos) / self.dt

        self.last_action = self.action
        self.action = delta_q + self.data.qpos[-12:]
        self.action = np.clip(self.action, a_min=self.lower_limits, a_max=self.upper_limits)
        self.torques = self._compute_torques(self.action)

        # compute torque
        self.do_simulation(self.torques, self.frame_skip)

        self.last_base_pos = self.base_pos
        self.last_base_vel = self.base_lin_vel
        self.last_feet_pos = self.feet_pos
        self.last_base_orientation = self.base_orientation
        self.last_dof_pos = self.dof_pos
        self.last_dof_vel = self.dof_vel
        
        self.base_pos = self.data.qpos[:3].copy()
        self.base_lin_vel = self.data.qvel[:3].copy()
        self.base_ang_vel = (self.last_base_orientation - self.base_orientation) / self.dt 
        self.feet_pos = self.data.geom_xpos.copy()
        self.base_orientation = self.base_rotation
        self.dof_pos = self.data.qpos[7:]
        self.dof_vel = (self.last_dof_pos - self.dof_pos) / self.dt
        self.torques = self._compute_torques(self.action)
        self.contact_forces = self.data.cfrc_ext

        # print("self.data.contact.geom2")
        # print(self.data.contact.geom2)
        # print("self.data.geom_xpos")
        # print(self.data.geom_xpos) 

        # feet_contact = self.data.contact.geom2
        terminate = self.terminated

        reward_lin_vel_z = self._reward_lin_vel_z() * self.cfg.reward_scale.lin_vel_z
        reward_ang_vel_xy = self._reward_ang_vel_xy() * self.cfg.reward_scale.ang_vel_xy
        reward_orientation = self._reward_orientation() * self.cfg.reward_scale.orientation
        reward_base_height = self._reward_base_height() * self.cfg.reward_scale.base_height
        reward_torques = self._reward_torques() * self.cfg.reward_scale.torques
        reward_dof_vel = self._reward_dof_vel() * self.cfg.reward_scale.dof_vel
        reward_dof_acc = self._reward_dof_acc() * self.cfg.reward_scale.dof_acc
        reward_action_rate = self._reward_action_rate() * self.cfg.reward_scale.action_rate      
        reward_collision = self._reward_collision() * self.cfg.reward_scale.collision        
        reward_termination = self._reward_termination() * self.cfg.reward_scale.termination      
        reward_dof_pos_limits = self._reward_dof_pos_limits() # * self.cfg.reward_scale.               
        reward_dof_vel_limits = self._reward_dof_vel_limits() # * self.cfg.reward_scale.                
        reward_torque_limits = self._reward_torque_limits() # * self.cfg.reward_scale.               
        reward_tracking_lin_vel = self._reward_tracking_lin_vel() * self.cfg.reward_scale.tracking_lin_vel 
        reward_tracking_ang_vel = self._reward_tracking_ang_vel() * self.cfg.reward_scale.tracking_ang_vel 
        reward_feet_air_time = self._reward_feet_air_time() * self.cfg.reward_scale.feet_air_time    
        reward_stumble = self._reward_stumble() * self.cfg.reward_scale.feet_stumble     
        reward_stand_still = self._reward_stand_still() * self.cfg.reward_scale.stand_still          
        reward_feet_contact_forces = self._reward_feet_contact_forces() # * self.cfg.reward_scale.               

        rewards = [
            reward_lin_vel_z,
            reward_ang_vel_xy,
            reward_orientation,
            reward_base_height,
            reward_torques,
            reward_dof_vel,
            reward_dof_acc,
            reward_action_rate,
            reward_collision,
            reward_termination,
            reward_dof_pos_limits,
            reward_dof_vel_limits,
            reward_torque_limits,
            reward_tracking_lin_vel,
            reward_tracking_ang_vel,
            reward_feet_air_time,
            reward_stumble,
            reward_stand_still,
            reward_feet_contact_forces
        ]

        if self.cfg.only_positive_rewards:
            rewards = np.clip(rewards, a_min=0.0, a_max=None)
        rewards_total = np.sum(rewards)

        observation = self._get_obs()

        info = {
            'reward_lin_vel_z': reward_lin_vel_z,
            'reward_ang_vel_xy': reward_ang_vel_xy,
            'reward_orientation': reward_orientation,
            'reward_base_height': reward_base_height,
            'reward_torques': reward_torques,
            'reward_dof_vel': reward_dof_vel,
            'reward_dof_acc': reward_dof_acc,
            'reward_action_rate': reward_action_rate,
            'reward_collision': reward_collision,
            'reward_termination': reward_termination,
            'reward_tracking_lin_vel': reward_tracking_lin_vel,
            'reward_tracking_ang_vel': reward_tracking_ang_vel,
            'reward_feet_air_time': reward_feet_air_time,
            'reward_stumble': reward_stumble,
            'reward_stand_still': reward_stand_still,
            'rewards_total': rewards_total,
            'traverse': self.data.qpos[0]
        }

        if self.render_mode == "human":
            self.render()
        return observation, rewards_total, terminate, info

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
            torques = self.cfg.control.stiffness*(actions_scaled - self.dof_vel) - self.cfg.control.damping*(self.dof_vel - self.last_dof_vel)/self.dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torques
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
from vehicle import Driver

MAX_STEER_ANGLE = 0.4
MIN_STEER_ANGLE = -0.4
MAX_SPEED = 150.0
MIN_SPEED = 0.0
MAX_SPEED_CHANGE = 75.0

MAX_SAFE_SPEED = 112.65  # ~ 70 mph
CITY_SPEED_LIMIT = 48.28  # ~ 30 mph

# MAX_SIM_TIME = 120  # 2 min max sim time
MAX_EPI_TIME = 60 # 1 min max episode time

GOAL_COORDS = [-36.75, 59.5]
GOAL_THRESHOLD = [1.75, 0.5]  # how close to goal to count as reached (in meters)


class WebotsCarEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 20}
    
    def __init__(self):
        super(WebotsCarEnv, self).__init__()
        
        self.agent = Driver()
        self.time_step = int(self.agent.getBasicTimeStep())
        self.start_time = None 
        
        # action space: [steering, speed]
        self.action_space = spaces.Box(
            low=np.array([MIN_STEER_ANGLE, MIN_SPEED]), 
            high=np.array([MAX_STEER_ANGLE, MAX_SPEED]),
            dtype=np.float32
        )
        
        # state space 
        self.state_space = spaces.Dict({
            "speed": spaces.Box(low=0, high=MAX_SPEED, shape=(1,), dtype=np.float32),  # gps speed
            "gps": spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),  # (x, y) gps coordinates
            "lidar_dist": spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32),  # distance to nearest obstacle
            "lidar_angle": spaces.Box(low=-90, high=90, shape=(1,), dtype=np.float32),  # angle to nearest obstacle
            "lane_deviation": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),  # pixels away from lane center
            "lane_mask": spaces.Box(low=0, high=1, shape=(64, 128, 1), dtype=np.uint8)  # binary mask for lane line (yellow line only)
        })
        
        # Initialize devices
        self.camera = self.agent.getDevice("camera")
        self.camera.enable(self.time_step)

        self.gps = self.agent.getDevice("gps")
        self.gps.enable(self.time_step)

        self.lidar = self.agent.getDevice("lidar")
        self.lidar.enable(self.time_step)
        
        self.gyro = self.agent.getDevice("gyro")
        self.gyro.enable(self.time_step)
        
        # Initialize state variables
        self.prev_gps_speed = 0.0
        self.gps_speed = 0.0
        self.gps_coords = [0.0, 0.0, 0.0]
        
        self.prev_set_speed = 0.0
        
        self.progress = 0.0
        
        # Lidar state variables
        self.lidar_dist = 100.0
        self.lidar_angle = 0.0
        self.collision_counter = 0
        
        self.reset_flag = self.agent.getFromDef("RESET_FLAG")
                        
                
    def step(self, action):
        steering_angle = action[0]
        speed = np.clip(action[1], MIN_SPEED, MAX_SPEED)
        
        self._set_steering_angle(steering_angle)
        self._set_cruising_speed(speed)
        self._calc_gps_speed()

        self.agent.step()

        state = self._get_state()
        reward = self._compute_reward()
        done = self._is_done()

        return state, reward, done
    
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.done = False
        self._collision_counter = 0  
        self.agent.setSteeringAngle(0)
        self.agent.setCruisingSpeed(0)
        self.reset_flag.getField("translation").setSFVec3f([1, 0, 0])  # send out flag to dummy node
        
        self.start_time = self.agent.getTime()
        
        for _ in range(5):  
            self.agent.step()

        state = self._get_state()

        return state
    
    
    def render(self, mode="human"):
        pass 
    
    
    def _get_state(self):
        speed = self.gps_speed
        position = self.gps.getValues() if self.gps else [0, 0, 0]
        
        # Process LIDAR data using the new helper function
        lidar_dist, lidar_angle = self._process_lidar()
        # Update internal lidar state variables
        self.lidar_dist = lidar_dist
        self.lidar_angle = lidar_angle
        
        frame = self._process_image()
        # cv2.imsthow("Frame", frame)
        if frame is not None:
            edges = self._create_lane_mask(frame)
            # cv2.imshow("Edges", edges)
        else:
            edges = np.zeros((64, 128), dtype=np.uint8)  # default blank image if no image available
        
        # cv2.waitKey(0)  # press 0 to close windows
        # cv2.destroyAllWindows()
        
        # Prevent NaN values in observations
        if any(np.isnan(position)) or np.isnan(speed):
            raise ValueError(f"Invalid observation values: speed={speed}, position={position}")

        _, lane_deviation = self._calc_lane_reward(k=1)
        print(f"lane deviation: {lane_deviation}")
        
        return {
            "speed": np.array([speed], dtype=np.float32),
            "gps": np.array([position[0], position[1]], dtype=np.float32),
            "lidar_dist": np.array([lidar_dist], dtype=np.float32), 
            "lidar_angle": np.array([lidar_angle], dtype=np.float32),
            "lane_deviation": np.array([lane_deviation], dtype=np.float32),
            "lane_mask": np.expand_dims(edges, axis=-1).astype(np.uint8)
        }
    
    def _compute_reward(self):
        reward = 0.0
        
        # Collision penalty
        if self._has_collided() or abs(self.gyro.getValues()[0]) > 0.5:
            reward -= 100
            
        lane_reward, _ = self._calc_lane_reward(k=1)
        reward += lane_reward
        
         # Speed efficiency reward          
        if 5 < self.gps_speed < CITY_SPEED_LIMIT:  # Favorable speed range
            reward += 15
        elif self.gps_speed > MAX_SAFE_SPEED:
            reward -= 20
        elif self.gps_speed >= CITY_SPEED_LIMIT:  # Penalty for speeding
            reward -= 10  
        elif self.gps_speed < 5:  # Penalty for stalling
            reward -= 30
        
        reward += self.gps_speed *.5  # speed progress reward
        
        # Distance-based reward    
        current_pos = np.linalg.norm(self.gps_coords[:2])
        progress_reward = max(0, current_pos - self.progress) * 500
        if current_pos > self.progress:
            reward += progress_reward
            print("progress reward: ", progress_reward)
        self.progress = current_pos
        
         
        # Obstacle avoidance penalty       
        if abs(self.lidar_angle) < 15 and self.lidar_dist < 2.0:
            extra_penalty = (2.0 - self.lidar_dist) * 100
            reward -= extra_penalty
        elif abs(self.lidar_angle) < 60:
            off_course_penalty = (90 - abs(self.lidar_angle))
            reward -= off_course_penalty
        
        if self._has_reached_goal():
            reward += 1000
            
        print(f"Total reward: {reward}")
        return reward
    
    
    def _is_done(self):
        if self._has_collided():
            print("Collision detected.")
            return True

        if self._has_reached_goal():
            print("Goal reached.")
            return True

        elapsed_time = self.agent.getTime() - self.start_time
        if elapsed_time >= MAX_EPI_TIME:
            print("Time limit reached.")
            return True
        
        if abs(self.gyro.getValues()[0]) > 0.5:
            print("Car flipped.")
            return True
            
        return False 
    
    
    def _has_collided(self):
        # LIDAR collision detection using central region of lidar scan
        if not self.lidar:
            return False

        lidar_data = self.lidar.getRangeImage()
        lidar_data = np.nan_to_num(lidar_data, nan=100.0, posinf=100.0, neginf=100.0)
        n = len(lidar_data)
        if n == 0:
            return False
        
        # Get lidar field-of-view (FOV) in radians and compute beam angles
        fov = self.lidar.getFov()  # in radians
        angles = np.linspace(-fov/2, fov/2, n)
        
        # Define thresholds
        collision_threshold = 1.0  # meters
        angle_threshold = np.radians(30)  # ±30° in radians
        min_beam_count = 3  # require at least 3 beams below threshold
        
        # Select beams within the central ±30° region
        central_indices = np.where(np.abs(angles) <= angle_threshold)[0]
        central_distances = lidar_data[central_indices]
        num_beams_below = np.sum(central_distances < collision_threshold)
        
        collision_detected = num_beams_below >= min_beam_count
        
        # Temporal consistency: require condition to persist for at least 2 consecutive steps
        if collision_detected:
            self._collision_counter += 1
        else:
            self._collision_counter = 0
        
        if self._collision_counter >= 2:
            print(f"Collision detected: {num_beams_below} beams below threshold in central region for {self._collision_counter} consecutive steps.")
            return True
        
        return False
    
    
    def _has_reached_goal(self):
        current_pos = self.gps_coords
        x_dist = abs(current_pos[0] - GOAL_COORDS[0]) < GOAL_THRESHOLD[0]
        y_dist = abs(current_pos[1] - GOAL_COORDS[1]) < GOAL_THRESHOLD[1]
    
        return x_dist and y_dist
        
        
    def _set_steering_angle(self, wheel_angle):
        steering_angle = self.agent.getSteeringAngle()
        if (wheel_angle - steering_angle > 0.1):
            wheel_angle = steering_angle + 0.1
        if (wheel_angle - steering_angle < -0.1):
            wheel_angle = steering_angle - 0.1
        steering_angle = wheel_angle
        
        if (wheel_angle > MAX_STEER_ANGLE):
            wheel_angle = MAX_STEER_ANGLE
        elif (wheel_angle < MIN_STEER_ANGLE):
            wheel_angle = MIN_STEER_ANGLE
            
        self.agent.setSteeringAngle(wheel_angle)
        return
    
    
    def _set_cruising_speed(self, cruising_speed):
        speed_diff = np.clip(cruising_speed - self.prev_set_speed, -MAX_SPEED_CHANGE, MAX_SPEED_CHANGE)
        smoothed_speed = self.prev_set_speed + speed_diff
        
        self.agent.setCruisingSpeed(smoothed_speed)
        self.prev_speed = smoothed_speed
        return
        
        
    def _calc_gps_speed(self):
        coords = self.gps.getValues()
        speed_ms = self.gps.getSpeed() * 3.6  # convert to km/h
        
        if coords is not None:
            self.prev_gps_speed = self.gps_speed 
            self.gps_speed = speed_ms 
            self.gps_coords = list(coords)
    
    
    def _calc_lane_reward(self, k=1.0):
        frame = self._process_image()
        edges = self._create_lane_mask(frame)
        left_lane = self._sliding_window_detect_lanes(edges)
        
        if not left_lane:
            penalty = -80
            deviation = 48 
        
        # avg of x pos of lane points closest to the car
        # x_right = max(right_lane, key=lambda pt: pt[1])[0]
        x_left = max(left_lane, key=lambda pt: pt[1])[0]
        
        x_vehicle = self.camera.getWidth() // 2
        lane_center = (x_left + 48)
        deviation = abs(x_vehicle - lane_center)
        
        penalty = k * -(deviation)
        
        if x_left == 0:
            penalty = -80
            deviation = 48
         
        if deviation < 5:
            penalty = 100 
        elif deviation < 10:
            penalty = 60
        elif deviation < 20:
            penalty = 40
        print("lane penalty: ", penalty)
        return penalty, deviation
        
      
    def _process_image(self):
        width = self.camera.getWidth()
        height = self.camera.getHeight()
        raw_image = self.camera.getImage()
        
        img = np.frombuffer(raw_image, np.uint8).reshape((height, width, 4))
        img_bgr = img[:, :, :3]  # remove alpha channel and only have BGR channels
        return img_bgr
    
    
    def _create_lane_mask(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # hue, saturation, value color filtering

        # yellow line detection
        lower_bound_yellow = np.array([18, 94, 140], dtype=np.uint8)
        upper_bound_yellow = np.array([48, 255, 255], dtype=np.uint8)
        yellow_mask = cv2.inRange(hsv, lower_bound_yellow, upper_bound_yellow)
        
        # white line detection
        # lower_bound_white = np.array([0, 0, 150], dtype=np.uint8)
        # upper_bound_white = np.array([180, 35, 255], dtype=np.uint8)
        # white_mask = cv2.inRange(hsv, lower_bound_white, upper_bound_white)
        
        # combined_mask = cv2.bitwise_or(yellow_mask, white_mask)

        # strengthen dashed lines
        # kernel = np.ones((3, 3), np.uint8)
        # combined_mask = cv2.dilate(combined_mask, kernel, iterations=2)
        
        height, width = frame.shape[:2]
        mask_roi = np.zeros_like(yellow_mask)
        
        # focus on bottom half of the frame
        roi_vertices = np.array([[
            (0, height), (width, height), (width, height//2), (0, height//2)
        ]], dtype=np.int32)

        cv2.fillPoly(mask_roi, roi_vertices, 255)
        combined_mask = cv2.bitwise_and(yellow_mask, mask_roi)  # apply region mask

        edges = cv2.Canny(combined_mask, 80, 150) 
        return edges
        
    
    # https://ieeexplore.ieee.org/document/9208278
    # https://www.youtube.com/watch?v=ApYo6tXcjjQ
    def _sliding_window_detect_lanes(self, edges):
        height, width = edges.shape
        
        # only get histogram for lower half of image where the lanes are
        histogram = np.sum(edges[height//2:, :], axis=0)
        
        midpoint = width // 2
        left_x_base = np.argmax(histogram[:midpoint])  # yellow lane line
        # right_x_base = np.argmax(histogram[midpoint:]) + midpoint # white dotted lane line
        
        # sliding window parameters
        num_windows = 10
        window_height = height // num_windows
        margin = 60
        min_pixels = 30
        
        left_x_current = left_x_base
        # right_x_current = right_x_base
        left_lane_pts = []
        # right_lane_pts = []
        
        for window in range(num_windows):
            # sliding window boundaries
            y_low = height - (window + 1) * window_height
            y_high = height - window * window_height

            x_left_low = int(left_x_current - margin)
            x_left_high = int(left_x_current + margin)

            # x_right_low = int(right_x_current - margin)
            # x_right_high = int(right_x_current + margin)
            
            # get lane pixels in each window
            left_lane_indices = np.where((edges[y_low:y_high, x_left_low:x_left_high] > 0))
            # right_lane_indices = np.where((edges[y_low:y_high, x_right_low:x_right_high] > 0))
            
            # yellow solid line
            if len(left_lane_indices[0]) > min_pixels:
                left_x_current = np.mean(left_lane_indices[1]) + x_left_low
                
            # white dotted line: check for lane pixels, if currently in gap use the previous window
            # if len(right_lane_indices[0]) > min_pixels:
            #     right_x_current = np.mean(right_lane_indices[1]) + x_right_low
            # elif len(right_lane_pts) > 0:
            #     right_x_current = right_lane_pts[-1][0]
                
            left_lane_pts.append((left_x_current, (y_low + y_high) // 2)) # yellow lane line
            # right_lane_pts.append((right_x_current, (y_low + y_high) // 2)) # white dotted lane line
            
        return left_lane_pts#, right_lane_pts
        
    
    def _process_lidar(self):
        """
        Processes the Webots LIDAR data to compute:
         - The minimum distance to an obstacle.
         - The angle (in degrees) corresponding to that measurement.
        """
        if not self.lidar:
            return 100.0, 0.0
        
        lidar_data = self.lidar.getRangeImage()
        # Replace NaN/infinite values with a safe maximum (assumed to be 100.0)
        lidar_data = np.nan_to_num(lidar_data, nan=100.0, posinf=100.0, neginf=0.0)
        if len(lidar_data) == 0:
            return 100.0, 0.0
        
        min_distance = np.min(lidar_data)
        min_index = np.argmin(lidar_data)
        
        fov = self.lidar.getFov()  # Field of view in radians
        n = len(lidar_data)
        # Calculate the angle corresponding to the minimum distance measurement.
        angle_rad = -fov / 2 + (min_index * fov / (n - 1))
        angle_deg = np.degrees(angle_rad)
        
        print(f"Lidar - min_distance: {min_distance:.2f}, angle_deg: {angle_deg:.2f}")
        return min_distance, angle_deg
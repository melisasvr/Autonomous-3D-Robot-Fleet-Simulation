import numpy as np
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import os
from pathlib import Path

# ----------------------------------------------------------------------
# OUTPUT CONFIGURATION – state is saved here
# ----------------------------------------------------------------------
OUTPUT_DIR = Path("./output")
OUTPUT_DIR.mkdir(exist_ok=True)
STATE_FILE = OUTPUT_DIR / "simulation_state.json"

# ----------------------------------------------------------------------
# Enums & data containers
# ----------------------------------------------------------------------
class RobotRole(Enum):
    SCOUT = "scout"
    CARRIER = "carrier"
    BLOCKER = "blocker"
    IDLE = "idle"

class HazardType(Enum):
    FIRE = "fire"
    BLOCKAGE = "blockage"
    CLEAR = "clear"

@dataclass
class Vector3:
    x: float
    y: float
    z: float
    
    def to_array(self):
        return np.array([self.x, self.y, self.z])
    
    def distance_to(self, other):
        diff = self.to_array() - other.to_array()
        return float(np.linalg.norm(diff))

@dataclass
class Hazard:
    position: Vector3
    hazard_type: HazardType
    severity: float
    id: str

@dataclass
class Building:
    position: Vector3
    size: Vector3
    
    def contains_point(self, point: Vector3) -> bool:
        """Check if a point is inside the building's bounding box"""
        half_x = self.size.x / 2
        half_z = self.size.z / 2
        
        return (abs(point.x - self.position.x) < half_x and
                abs(point.z - self.position.z) < half_z)
    
    def get_nearest_edge_point(self, point: Vector3) -> Vector3:
        """Get the nearest point on the building's edge to push robot out"""
        half_x = self.size.x / 2
        half_z = self.size.z / 2
        
        left  = abs((self.position.x - half_x) - point.x)
        right = abs((self.position.x + half_x) - point.x)
        front = abs((self.position.z - half_z) - point.z)
        back  = abs((self.position.z + half_z) - point.z)
        
        min_dist = min(left, right, front, back)
        
        if min_dist == left:
            return Vector3(self.position.x - half_x - 1, point.y, point.z)
        elif min_dist == right:
            return Vector3(self.position.x + half_x + 1, point.y, point.z)
        elif min_dist == front:
            return Vector3(point.x, point.y, self.position.z - half_z - 1)
        else:
            return Vector3(point.x, point.y, self.position.z + half_z + 1)

# ----------------------------------------------------------------------
# City generation
# ----------------------------------------------------------------------
class CityGenerator:
    def __init__(self, grid_size=10, block_size=20):
        self.grid_size = grid_size
        self.block_size = block_size
        self.buildings_objects = []
        
    def generate_city(self):
        buildings = []
        roads = []
        self.buildings_objects = []
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if (i + j) % 2 == 0:
                    x = float(i * self.block_size)
                    z = float(j * self.block_size)
                    height = float(np.random.uniform(10, 40))
                    
                    building_dict = {
                        'position': {'x': x, 'y': height/2, 'z': z},
                        'size': {'x': self.block_size * 0.8, 'y': height, 'z': self.block_size * 0.8}
                    }
                    buildings.append(building_dict)
                    
                    self.buildings_objects.append(Building(
                        position=Vector3(x, height/2, z),
                        size=Vector3(self.block_size * 0.8, height, self.block_size * 0.8)
                    ))
                else:
                    x = float(i * self.block_size)
                    z = float(j * self.block_size)
                    roads.append({
                        'position': {'x': x, 'y': 0.1, 'z': z},
                        'size': {'x': float(self.block_size), 'y': 0.2, 'z': float(self.block_size)}
                    })
        
        return {'buildings': buildings, 'roads': roads}
    
    def generate_hazards(self, num_hazards=5):
        hazards = []
        for i in range(num_hazards):
            attempts = 0
            while attempts < 50:
                x = float(np.random.uniform(0, self.grid_size * self.block_size))
                z = float(np.random.uniform(0, self.grid_size * self.block_size))
                test_pos = Vector3(x, 0, z)
                
                if not any(building.contains_point(test_pos) for building in self.buildings_objects):
                    break
                attempts += 1
            
            hazard_type = np.random.choice([HazardType.FIRE, HazardType.BLOCKAGE])
            
            hazards.append(Hazard(
                position=Vector3(x, 0, z),
                hazard_type=hazard_type,
                severity=float(np.random.uniform(0.5, 1.0)),
                id=f"hazard_{i}"
            ))
        
        return hazards

# ----------------------------------------------------------------------
# Tiny neural network (no training – just random policy)
# ----------------------------------------------------------------------
class NeuralNetwork:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.w1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.w2 = np.random.randn(hidden_dim, output_dim) * 0.1
        self.b2 = np.zeros(output_dim)
        
    def forward(self, x):
        h = np.maximum(0, np.dot(x, self.w1) + self.b1)
        return np.dot(h, self.w2) + self.b2
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

# ----------------------------------------------------------------------
# Robot agent
# ----------------------------------------------------------------------
class RobotAgent:
    def __init__(self, agent_id: int, position: Vector3, robot_type: str):
        self.id = agent_id
        self.position = position
        self.velocity = Vector3(0, 0, 0)
        self.robot_type = robot_type
        self.role = RobotRole.IDLE
        self.communication_vector = np.random.randn(16)
        
        self.radius = 2.0 if robot_type == "wheeled" else 1.5
        
        self.policy_net = NeuralNetwork(input_dim=32, hidden_dim=64, output_dim=8)
        self.role_net   = NeuralNetwork(input_dim=26, hidden_dim=32, output_dim=4)
        
        self.target_position = None
        self.detected_hazards = []
        self.energy = 100.0
        self.rewards = []
        self.collision_count = 0
        
    def observe_environment(self, hazards: List[Hazard], other_agents: List['RobotAgent']):
        obs = []
        
        # Position (3)
        obs.extend([self.position.x, self.position.y, self.position.z])
        # Velocity (3)
        obs.extend([self.velocity.x, self.velocity.y, self.velocity.z])
        # Energy (1)
        obs.append(self.energy / 100.0)
        
        # Role one-hot (4)
        obs.append(float(self.role.value == RobotRole.SCOUT.value))
        obs.append(float(self.role.value == RobotRole.CARRIER.value))
        obs.append(float(self.role.value == RobotRole.BLOCKER.value))
        obs.append(float(self.role.value == RobotRole.IDLE.value))
        
        # Type one-hot (3)
        obs.append(float(self.robot_type == "wheeled"))
        obs.append(float(self.robot_type == "legged"))
        obs.append(float(self.robot_type == "hybrid"))
        
        # Hazard data (3 hazards * 4 values = 12)
        nearest_hazards = sorted(hazards, key=lambda h: self.position.distance_to(h.position))[:3]
        for hazard in nearest_hazards:
            dist = self.position.distance_to(hazard.position)
            obs.extend([
                (hazard.position.x - self.position.x) / 100.0,
                (hazard.position.z - self.position.z) / 100.0,
                dist / 100.0,
                hazard.severity
            ])
        for _ in range(3 - len(nearest_hazards)):
            obs.extend([0, 0, 0, 0])
        
        # Other agent data (2 agents * 3 values = 6)
        nearest_agents = sorted(other_agents, key=lambda a: self.position.distance_to(a.position))[:2]
        for agent in nearest_agents:
            dist = self.position.distance_to(agent.position)
            obs.extend([
                (agent.position.x - self.position.x) / 100.0,
                (agent.position.z - self.position.z) / 100.0,
                dist / 100.0
            ])
        for _ in range(2 - len(nearest_agents)):
            obs.extend([0, 0, 0])
        
        return np.array(obs[:32])
    
    def select_role(self, observation):
        role_obs = observation[:26]
        role_logits = self.role_net.forward(role_obs)
        role_probs = self.role_net.softmax(role_logits)
        
        role_idx = np.random.choice(4, p=role_probs)
        roles = [RobotRole.SCOUT, RobotRole.CARRIER, RobotRole.BLOCKER, RobotRole.IDLE]
        self.role = roles[role_idx]
        return self.role
    
    def select_action(self, observation):
        action_logits = self.policy_net.forward(observation)
        action_probs = self.policy_net.softmax(action_logits)
        
        action_idx = np.random.choice(8, p=action_probs)
        
        speed = 0.5 if self.robot_type == "wheeled" else 0.3
        actions = [
            (speed, 0), (-speed, 0),
            (0, speed), (0, -speed),
            (speed, speed), (speed, -speed),
            (-speed, speed), (-speed, -speed)
        ]
        return actions[action_idx]
    
    def update_communication_vector(self, other_agents: List['RobotAgent']):
        nearby = [a for a in other_agents if self.position.distance_to(a.position) < 30]
        if nearby:
            avg_comm = np.mean([a.communication_vector for a in nearby], axis=0)
            self.communication_vector = 0.8 * self.communication_vector + 0.2 * avg_comm
            norm = np.linalg.norm(self.communication_vector)
            if norm > 1e-8:
                self.communication_vector = self.communication_vector / norm
    
    def move(self, action: Tuple[float, float], dt: float = 0.016):
        dx, dz = action
        
        self.velocity.x = dx
        self.velocity.z = dz
        
        old_x = self.position.x
        old_z = self.position.z
        
        self.position.x += self.velocity.x * dt * 100
        self.position.z += self.velocity.z * dt * 100
        
        self.position.x = float(np.clip(self.position.x, 0, 200))
        self.position.z = float(np.clip(self.position.z, 0, 200))
        
        self.energy -= 0.1 * dt
    
    def check_building_collision(self, buildings: List[Building]) -> bool:
        for building in buildings:
            if building.contains_point(self.position):
                edge_point = building.get_nearest_edge_point(self.position)
                self.position.x = edge_point.x
                self.position.z = edge_point.z
                self.velocity.x = 0
                self.velocity.z = 0
                self.collision_count += 1
                return True
        return False
    
    def check_agent_collision(self, other_agents: List['RobotAgent']) -> bool:
        for other in other_agents:
            dist = self.position.distance_to(other.position)
            min_dist = self.radius + other.radius
            
            if dist < min_dist and dist > 0.01:
                overlap = min_dist - dist
                dx = self.position.x - other.position.x
                dz = self.position.z - other.position.z
                length = np.sqrt(dx*dx + dz*dz)
                
                if length > 0.01:
                    dx /= length
                    dz /= length
                    
                    self.position.x += dx * overlap * 0.5
                    self.position.z += dz * overlap * 0.5
                    other.position.x -= dx * overlap * 0.5
                    other.position.z -= dz * overlap * 0.5
                    
                    self.velocity.x *= 0.5
                    self.velocity.z *= 0.5
                    other.velocity.x *= 0.5
                    other.velocity.z *= 0.5
                    
                    self.collision_count += 1
                    return True
        return False
    
    def detect_hazards(self, hazards: List[Hazard], detection_radius=20):
        self.detected_hazards = [
            h for h in hazards
            if self.position.distance_to(h.position) < detection_radius
        ]
    
    def compute_reward(self, hazards: List[Hazard]):
        reward = 0.0
        
        if len(self.detected_hazards) > 0:
            reward += 5.0 * len(self.detected_hazards)
        
        if self.role == RobotRole.SCOUT:
            reward += 2.0 if len(self.detected_hazards) > 0 else -0.5
        elif self.role == RobotRole.CARRIER:
            reward += 1.0
        elif self.role == RobotRole.BLOCKER:
            reward += 1.0
        
        reward -= 0.1
        
        if self.energy < 20:
            reward -= 2.0
        
        if self.collision_count > 0:
            reward -= 1.0
        
        self.rewards.append(reward)
        return reward

# ----------------------------------------------------------------------
# Multi-agent system
# ----------------------------------------------------------------------
class MultiAgentSystem:
    def __init__(self, num_agents=5, grid_size=10):
        self.city_gen = CityGenerator(grid_size=grid_size)
        self.city = self.city_gen.generate_city()
        self.hazards = self.city_gen.generate_hazards(num_hazards=8)
        
        self.agents = []
        types = ["wheeled", "legged", "hybrid"]
        
        for i in range(num_agents):
            attempts = 0
            while attempts < 100:
                x = float(np.random.uniform(10, 190))
                z = float(np.random.uniform(10, 190))
                test_pos = Vector3(x, 0, z)
                if not any(b.contains_point(test_pos) for b in self.city_gen.buildings_objects):
                    break
                attempts += 1
            
            robot_type = types[i % 3]
            agent = RobotAgent(i, Vector3(x, 0, z), robot_type)
            self.agents.append(agent)
        
        self.timestep = 0
        self.global_stats = {
            'total_hazards_detected': 0,
            'avg_reward': 0.0,
            'role_distribution': {},
            'total_collisions': 0
        }
    
    def step(self):
        for agent in self.agents:
            agent.collision_count = 0
        
        for agent in self.agents:
            obs = agent.observe_environment(self.hazards, [a for a in self.agents if a.id != agent.id])
            
            if self.timestep % 50 == 0:
                agent.select_role(obs)
            
            action = agent.select_action(obs)
            agent.move(action)
        
        for agent in self.agents:
            agent.check_building_collision(self.city_gen.buildings_objects)
            agent.check_agent_collision([a for a in self.agents if a.id != agent.id])
        
        for agent in self.agents:
            agent.detect_hazards(self.hazards)
            agent.update_communication_vector([a for a in self.agents if a.id != agent.id])
            agent.compute_reward(self.hazards)
        
        self.timestep += 1
        self.update_stats()
    
    def update_stats(self):
        total_detected = sum(len(a.detected_hazards) for a in self.agents)
        self.global_stats['total_hazards_detected'] = total_detected
        
        if self.agents and any(a.rewards for a in self.agents):
            avg_reward = np.mean([a.rewards[-1] for a in self.agents if a.rewards])
            self.global_stats['avg_reward'] = float(avg_reward)
        
        role_counts = {}
        for a in self.agents:
            role = a.role.value
            role_counts[role] = role_counts.get(role, 0) + 1
        self.global_stats['role_distribution'] = role_counts
        
        total_collisions = sum(a.collision_count for a in self.agents)
        self.global_stats['total_collisions'] = int(total_collisions)
    
    def get_state(self):
        return {
            'city': self.city,
            'hazards': [{
                'position': {'x': float(h.position.x), 'y': float(h.position.y), 'z': float(h.position.z)},
                'type': h.hazard_type.value,
                'severity': float(h.severity),
                'id': h.id
            } for h in self.hazards],
            'agents': [{
                'id': int(a.id),
                'position': {'x': float(a.position.x), 'y': float(a.position.y), 'z': float(a.position.z)},
                'velocity': {'x': float(a.velocity.x), 'y': float(a.velocity.y), 'z': float(a.velocity.z)},
                'role': a.role.value,
                'type': a.robot_type,
                'energy': float(a.energy),
                'detected_hazards': len(a.detected_hazards),
                'communication': [float(x) for x in a.communication_vector.tolist()[:4]],
                'collision_count': int(a.collision_count)
            } for a in self.agents],
            'stats': {
                'total_hazards_detected': int(self.global_stats['total_hazards_detected']),
                'avg_reward': float(self.global_stats['avg_reward']),
                'role_distribution': {k: int(v) for k, v in self.global_stats['role_distribution'].items()},
                'total_collisions': int(self.global_stats['total_collisions'])
            },
            'timestep': int(self.timestep)
        }
    
    def to_json(self):
        return json.dumps(self.get_state(), indent=4)

# ----------------------------------------------------------------------
# Global system (lazy init)
# ----------------------------------------------------------------------
mas_system = None

def get_system():
    global mas_system
    if mas_system is None:
        mas_system = MultiAgentSystem(num_agents=7, grid_size=10)
    return mas_system

# ----------------------------------------------------------------------
# Public API – now also writes the JSON to disk
# ----------------------------------------------------------------------
def step_simulation():
    """Advance simulation by one step, return JSON and write it to output/simulation_state.json"""
    try:
        system = get_system()
        system.step()
        state_json = system.to_json()
        STATE_FILE.write_text(state_json, encoding="utf-8")
        return state_json
    except Exception as e:
        err = json.dumps({'error': str(e), 'type': type(e).__name__})
        STATE_FILE.write_text(err, encoding="utf-8")
        return err

def reset_simulation():
    """Reset the simulation and write the fresh initial state."""
    global mas_system
    try:
        mas_system = MultiAgentSystem(num_agents=7, grid_size=10)
        state_json = mas_system.to_json()
        STATE_FILE.write_text(state_json, encoding="utf-8")
        return state_json
    except Exception as e:
        err = json.dumps({'error': str(e), 'type': type(e).__name__})
        STATE_FILE.write_text(err, encoding="utf-8")
        return err

def get_current_state():
    """Return current JSON (does NOT step) – also writes it."""
    try:
        state_json = get_system().to_json()
        STATE_FILE.write_text(state_json, encoding="utf-8")
        return state_json
    except Exception as e:
        err = json.dumps({'error': str(e), 'type': type(e).__name__})
        STATE_FILE.write_text(err, encoding="utf-8")
        return err

def set_num_agents(num_agents):
    """Create a fresh simulation with a custom number of agents and write the state."""
    global mas_system
    try:
        mas_system = MultiAgentSystem(num_agents=num_agents, grid_size=10)
        state_json = mas_system.to_json()
        STATE_FILE.write_text(state_json, encoding="utf-8")
        return state_json
    except Exception as e:
        err = json.dumps({'error': str(e), 'type': type(e).__name__})
        STATE_FILE.write_text(err, encoding="utf-8")
        return err

# ----------------------------------------------------------------------
# Standalone test (unchanged except it now also leaves a file)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("--- Initializing Simulation with Collision Detection ---")
    initial_state = get_current_state()
    print(initial_state)
    
    print("\n--- Running 10 Simulation Steps ---")
    for i in range(10):
        print(f"Stepping... {i+1}/10")
        result = step_simulation()
        state = json.loads(result)
        if state.get('stats'):
            print(f"  Collisions this step: {state['stats'].get('total_collisions', 0)}")
    
    print("\n--- Final State ---")
    final_state = get_current_state()
    state = json.loads(final_state)
    print(f"Total collisions: {state['stats']['total_collisions']}")
    print(f"Timestep: {state['timestep']}")
    print("\nThe latest state is also saved to: ", STATE_FILE)


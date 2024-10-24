'''
made by ChatGPT, edited by Vikhyat
'''

import numpy as np
import random
import math

class Node:
    def __init__(self, config, parent=None, cost=0.0):
        self.config = config  # Configuration (x, y, θ₁, θ₂, θ₃, θ₄)
        self.parent = parent  # Pointer to the parent node
        self.cost = cost      # Cost from the root to this node (for RRT*)

class RRTStar:
    def __init__(self, start, goal, minCoords, maxCoords, goal_test, max_iterations=1000, step_size=0.5, 
                goal_bias=0.05, search_radius=1.0, collisionChecker=(lambda x: False), verbose=0, seed=0):
        self.start = Node(start)  # start configuration
        self.goal = Node(goal)    # approximate goal configuration to sample with probability [goal_bias]
        self.minCoords = minCoords
        self.maxCoords = maxCoords
        self.goal_test = goal_test # function to test if a given state is a goal state

        self.max_iterations = max_iterations
        self.step_size = step_size
        self.goal_bias = goal_bias
        self.search_radius = search_radius
        self.has_collision = collisionChecker # Collision checker function (returns True if collision at a config)
        self.tree = [self.start]  # List to store the nodes of the tree
        self.verbose=verbose
        np.random.seed(seed)

    # Euclidean distance between two configurations
    def distance(self, q1, q2):
        return np.linalg.norm(np.array(q1) - np.array(q2))

    # Function to randomly sample from the configuration space
    def sample_random_configuration(self):
        if np.random.rand() < self.goal_bias:
            return self.goal.config
        return np.random.uniform(low=self.minCoords, high=self.maxCoords)

    # Find the nearest node in the tree to a given configuration
    def get_nearest_neighbor(self, config):
        nearest_node = None
        min_dist = float('inf')
        for node in self.tree:
            dist = self.distance(node.config, config)
            if dist < min_dist:
                nearest_node = node
                min_dist = dist
        return nearest_node

    # Steer from a config towards another config by atmost step_size distance
    def steer(self, configA, configB):
        direction = configB - configA
        length = np.linalg.norm(direction)
        if length < self.step_size:
            if self.verbose and abs(configB[0]-0.77625166)<0.001: print(f'pre-approved {configB} from {configA}')
            return configB
        newConfig = configA + (direction / length) * self.step_size
        if self.verbose and abs(newConfig[0]-0.77625166)<0.001: print(f'adjust-n-approved {newConfig} from {configA}')
        # print(f'steered: AB={newConfig-configA}, AR={configB-configA}')
        return newConfig

    # Find all nodes within the search radius of a given config
    def get_nearby_nodes(self, q):
        return [node for node in self.tree if self.distance(node.config, q) < self.search_radius]

    # Rewire tree by checking if connecting nearby nodes through the new node gives a lower cost path for them
    def rewire(self, new_node, nearby_nodes):
        for node in nearby_nodes:
            new_cost = new_node.cost + self.distance(new_node.config, node.config)
            if new_cost < node.cost and not self.has_collision(node.config):
                node.parent = new_node
                node.cost = new_cost

    # Function to trace the path from start to goal by traversing the tree from goal node
    def get_path_to_goal(self):
        path = []
        node = self.goal
        while node is not None:
            path.append(node.config)
            node = node.parent
        return path[::-1]  # Return path from start to goal

    # RRT* Algorithm
    def run_rrt_star(self):
        for i in range(self.max_iterations):
            # Sample a random configuration
            q_rand = self.sample_random_configuration()

            # Find the nearest node in the tree
            nearest_node = self.get_nearest_neighbor(q_rand)

            # Steer from the nearest node towards the random sample
            q_new_config = self.steer(nearest_node.config, q_rand)

            # If the new configuration has a collision, proceed
            if self.has_collision(q_new_config):
                continue

            # Create a new node and set its parent as nearest_node
            new_node = Node(q_new_config, nearest_node)
            new_node.cost = nearest_node.cost + self.distance(nearest_node.config, q_new_config)
            
            # Find nearby nodes within a certain radius to connect to the new node
            nearby_nodes = self.get_nearby_nodes(new_node.config)

            # dx_ratio = (new_node.config-nearest_node.config)[0]/(q_rand-nearest_node.config)[0]
            # dy_ratio = (new_node.config-nearest_node.config)[1]/(q_rand-nearest_node.config)[1]
            # assert abs(dx_ratio-dy_ratio)<1e-3, f'{dx_ratio} mismatch {dy_ratio}'
        
            # Choose the parent with minimum cost to reach new_node
            min_cost_node = nearest_node
            min_cost = new_node.cost
            for node in nearby_nodes:
                cost = node.cost + self.distance(node.config, new_node.config)
                if cost < min_cost and not self.has_collision(node.config):
                    min_cost_node = node
                    min_cost = cost
            new_node.parent = min_cost_node
            new_node.cost = min_cost

            # Add the new node to the tree
            self.tree.append(new_node)

            # # Rewire the nearby nodes to ensure optimality
            self.rewire(new_node, nearby_nodes)

            # Check if the new node is close enough to the goal
            if self.goal_test(new_node.config):
                self.goal.parent = new_node
                self.goal.cost = new_node.cost
                print(f"Goal reached in {i+1} iterations.")
                return self.get_path_to_goal()
        
        print("No valid path found.")
        return None

# Example usage
if __name__ == "__main__":
    start_config = np.array([0, 0, 0, 0, 0, 0])  # Initial configuration
    goal_config = np.array([4, 4, np.pi/4, np.pi/3, -np.pi/6, np.pi/2])  # Goal configuration
    
    rrt_star = RRTStar(start=start_config, goal=goal_config,
    minCoords=[0,0,0,0,0,0], maxCoords=[7,7,7,7,7,7], goal_test=lambda x:np.sum(np.abs(x-goal_config))<0.1)
    path = rrt_star.run_rrt_star()

    start_config = np.array([0, 0])  # Initial configuration
    goal_config = np.array([4, np.pi/2])  # Goal configuration
    
    rrt_star = RRTStar(start=start_config, goal=goal_config,
    minCoords=[0,0], maxCoords=[7,7], goal_test=lambda x:np.sum(np.abs(x-goal_config))<0.1)
    path = rrt_star.run_rrt_star()

    if path:
        print("Found path:")
        for config in path:
            print(config)
    else:
        print("Path not found.")

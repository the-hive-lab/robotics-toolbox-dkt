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
    def __init__(self, start, goal, minCoords, maxCoords, max_iterations=1000, step_size=0.5, goal_threshold=0.5, search_radius=1.0):
        self.start = Node(start)  # Start configuration node
        self.goal = Node(goal)    # Goal configuration node
        self.minCoords = minCoords
        self.maxCoords = maxCoords
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.goal_threshold = goal_threshold
        self.search_radius = search_radius
        self.tree = [self.start]  # List to store the nodes of the tree

    # Euclidean distance between two configurations
    def distance(self, q1, q2):
        return np.linalg.norm(np.array(q1) - np.array(q2))

    # Function to randomly sample from the configuration space
    def sample_random_configuration(self, goal_bias=0.05):
        if random.random() < goal_bias:
            return self.goal.config
        return np.random.uniform(low=self.minCoords,high=self.maxCoords)

    # Find the nearest node in the tree to a random configuration
    def nearest_neighbor(self, q_rand):
        nearest_node = None
        min_dist = float('inf')
        for node in self.tree:
            dist = self.distance(node.config, q_rand)
            if dist < min_dist:
                nearest_node = node
                min_dist = dist
        return nearest_node

    # Steer from the nearest node toward the random configuration
    def steer(self, q_nearest, q_rand):
        q_nearest = np.array(q_nearest)
        q_rand = np.array(q_rand)
        direction = q_rand - q_nearest
        length = np.linalg.norm(direction)
        if length < self.step_size:
            return q_rand.tolist()
        else:
            new_config = q_nearest + (direction / length) * self.step_size
            return new_config.tolist()

    # Collision checker function (placeholder: returns True if no obstacle)
    def obstacle_free(self, q):
        return True

    # Find all nodes within a certain radius of a new node
    def near(self, q_new):
        nearby_nodes = []
        for node in self.tree:
            if self.distance(node.config, q_new.config) < self.search_radius:
                nearby_nodes.append(node)
        return nearby_nodes

    # Rewire tree by checking if connecting nearby nodes through the new node gives a lower cost path
    def rewire(self, new_node, nearby_nodes):
        for node in nearby_nodes:
            new_cost = new_node.cost + self.distance(new_node.config, node.config)
            if new_cost < node.cost and self.obstacle_free(node.config):
                node.parent = new_node
                node.cost = new_cost

    # Function to trace the path from start to goal by traversing the tree from goal node
    def path_to_goal(self, goal_node):
        path = []
        node = goal_node
        while node is not None:
            path.append(node.config)
            node = node.parent
        return path[::-1]  # Return path from start to goal

    # RRT* Algorithm
    def rrt_star(self):
        for i in range(self.max_iterations):
            # Sample a random configuration
            q_rand = self.sample_random_configuration()

            # Find the nearest node in the tree
            nearest_node = self.nearest_neighbor(q_rand)

            # Steer from the nearest node towards the random sample
            q_new_config = self.steer(nearest_node.config, q_rand)

            # If the new configuration is obstacle-free, proceed
            if self.obstacle_free(q_new_config):
                # Create a new node and set its parent as nearest_node
                new_node = Node(q_new_config, nearest_node)
                new_node.cost = nearest_node.cost + self.distance(nearest_node.config, q_new_config)
                
                # Find nearby nodes within a certain radius to connect to the new node
                nearby_nodes = self.near(new_node)

                # Choose the parent with minimum cost to reach new_node
                min_cost_node = nearest_node
                min_cost = new_node.cost
                for node in nearby_nodes:
                    cost = node.cost + self.distance(node.config, new_node.config)
                    if cost < min_cost and self.obstacle_free(node.config):
                        min_cost_node = node
                        min_cost = cost
                new_node.parent = min_cost_node
                new_node.cost = min_cost

                # Add the new node to the tree
                self.tree.append(new_node)

                # Rewire the nearby nodes to ensure optimality
                self.rewire(new_node, nearby_nodes)

                # Check if the new node is close enough to the goal
                if self.distance(new_node.config, self.goal.config) < self.goal_threshold:
                    self.goal.parent = new_node
                    self.goal.cost = new_node.cost
                    print(f"Goal reached in {i+1} iterations.")
                    return self.path_to_goal(self.goal)
        
        print("No valid path found.")
        return None

# Example usage
if __name__ == "__main__":
    start_config = [0, 0, 0, 0, 0, 0]  # Initial configuration
    goal_config = [4, 4, np.pi/4, np.pi/3, -np.pi/6, np.pi/2]  # Goal configuration
    
    rrt_star = RRTStar(start=start_config, goal=goal_config)
    path = rrt_star.rrt_star()

    if path:
        print("Found path:")
        for config in path:
            print(config)
    else:
        print("Path not found.")

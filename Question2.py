import itertools
import math
import random

type_arrays = {"all": []}
# dictionary for saving all the points divided by there type
VIRGINICA = 1  # "Iris-virginica"
VERSICOLOR = 0  # "Iris-versicolor"

type_num_dic = {"Iris-virginica": VIRGINICA,
                "Iris-versicolor": VERSICOLOR}


class Point:
    def __init__(self, x, y, t):
        self.x = x
        self.y = y
        self.type = t
        self.weight = 0

    def __repr__(self):
        return f"({self.x}, {self.y})"

    def getPoint(self):
        return [self.x, self.y]

    def getType(self):
        return self.type

    def setWeight(self, w):
        self.weight = w

    def getWeight(self):
        return self.weight

    def getX(self):
        return self.x

    def getY(self):
        return self.y


def process_file(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.split()
            if len(parts) < 5:
                continue  # Skip invalid lines

            type_name = parts[4]
            if type_name == "Iris-versicolor" or type_name == "Iris-virginica":
                x, y = float(parts[1]), float(parts[2])

                # Create a point
                point = Point(x, y, type_num_dic.get(type_name))

                # Add the point to the appropriate array
                if type_name not in type_arrays:
                    type_arrays[type_name] = []
                type_arrays[type_name].append(point)
                type_arrays["all"].append(point)

    return type_arrays


class Node:
    def __init__(self, points, is_root, max_level=0):
        self.points = points
        self.left = None
        self.right = None
        self.split_feature = None
        self.split_value = None
        self.label = None
        self.max_level = max_level
        self.is_root = is_root

    def is_leaf(self):
        if self.is_root:
            return False
        return self.left is None and self.right is None

    def assign_label(self):
        if not self.points:  # If no points, assign a default label
            self.label = 0  # Default to 0 or some other convention
            return
        labels = [point.getType() for point in self.points]
        self.label = max(set(labels), key=labels.count)

    def calculate_error(self):
        """
        Calculate the classification error for this node.
        Error is the number of misclassified points if all points are labeled with this node's label.
        """
        if not self.points:
            return 0  # No points, no error

        if self.label is None:
            self.assign_label()  # Ensure the node has a label

        # Count misclassified points
        misclassified = sum(1 for point in self.points if point.getType() != self.label)
        return misclassified

    def split(self, feature, value):
        if feature == 'x':
            left_points = [p for p in self.points if p.getX() <= value]
            right_points = [p for p in self.points if p.getX() > value]
        elif feature == 'y':
            left_points = [p for p in self.points if p.getY() <= value]
            right_points = [p for p in self.points if p.getY() > value]
        else:
            raise ValueError("Invalid feature for splitting.")

        self.left = Node(left_points, False)
        self.right = Node(right_points, False)
        self.split_feature = feature
        self.split_value = value


class DecisionTree:
    def __init__(self, points, max_level):
        self.points = points
        self.max_level = max_level
        self.root = Node(points, True, max_level=max_level)

    def test_tree(self, tree, test_set):
        misclassified = 0
        for point in test_set:
            predicted_label = self.predict(tree, point)  # Predict the label for each point
            if predicted_label != point.getType():  # Compare with true label
                misclassified += 1
        return misclassified / len(test_set)  # Return the error rate as misclassified points / total points

    def predict(self, tree, point):
        current_node = tree
        while not current_node.is_leaf():
            if current_node.split_feature == 'x':
                if point.getX() <= current_node.split_value:
                    current_node = current_node.left
                else:
                    current_node = current_node.right
            elif current_node.split_feature == 'y':
                if point.getY() <= current_node.split_value:
                    current_node = current_node.left
                else:
                    current_node = current_node.right
        return current_node.label

    def build_tree(self, node, split_plan, values, depth=0):
        # Ensure the tree stops building if depth reaches max_level
        if depth >= self.max_level or not node.points:
            node.assign_label()  # Assign label if leaf or max depth reached
            return node

        # Get the split feature and value for the current depth
        feature = split_plan[depth]
        value = values[depth]

        # Split the current node
        node.split(feature, value)

        # Recursively build left and right subtrees
        self.build_tree(node.left, split_plan, values, depth + 1)
        self.build_tree(node.right, split_plan, values, depth + 1)

        return node

    def brute_force(self):
        # Brute-force tree construction (strategy A)
        best_tree = None
        min_error = float('inf')

        # Generate all possible trees up to max_level
        for split_plan in itertools.product(['x', 'y'], repeat=self.max_level):

            # Create a list of X and Y values
            x_values = [p.getX() for p in self.points]
            y_values = [p.getY() for p in self.points]
            # Combine X and Y values
            combined_values = x_values + y_values

            # Generate permutations of the combined values with the length of split_plan
            for values in itertools.permutations(combined_values, len(split_plan)):

                tree = self.build_tree(self.root, split_plan, values)
                tree.assign_label()
                error = self.calculate_tree_error(tree)
                if error < min_error:
                    min_error = error
                    best_tree = tree

        return best_tree, min_error

    def binary_entropy(self):
        # Binary entropy-based tree construction (strategy B)
        def entropy(points):
            labels = [point.getType() for point in points]
            total = len(labels)
            if total == 0:
                return 0
            prob_1 = labels.count(1) / total
            prob_2 = labels.count(-1) / total
            return -sum(p * math.log2(p) for p in [prob_1, prob_2] if p > 0)

        def best_split(node):
            best_feature, best_value, min_entropy = None, None, float('inf')
            for feature in ['x', 'y']:
                for point in node.points:
                    value = point.getX() if feature == 'x' else point.getY()
                    left_points = [p for p in node.points if (p.getX() if feature == 'x' else p.getY()) <= value]
                    right_points = [p for p in node.points if (p.getX() if feature == 'x' else p.getY()) > value]
                    entropy_sum = len(left_points) * entropy(left_points) + len(right_points) * entropy(right_points)
                    if entropy_sum < min_entropy:
                        best_feature, best_value, min_entropy = feature, value, entropy_sum
            return best_feature, best_value

        def recursive_split(node, depth):
            # Stop recursion if max depth is reached or node is already a leaf
            if depth >= self.max_level:
                node.assign_label()
                return

            feature, value = best_split(node)
            if feature is None:  # If no valid split, assign label and stop
                node.assign_label()
                return

            # Perform the split
            node.split(feature, value)

            # Recursively split left and right subtrees
            recursive_split(node.left, depth + 1)
            recursive_split(node.right, depth + 1)

        recursive_split(self.root, 0)  # Start from the root at depth 0
        return self.root

    def calculate_tree_error(self, node):
        if node.is_leaf():
            return node.calculate_error()  # Calculate error for leaf nodes
        return self.calculate_tree_error(node.left) + self.calculate_tree_error(node.right)

    def draw_tree(self, node, depth=0, prefix="", side="Root"):
        # Visualize the tree structure with graphical approach and side indicators
        if node.is_leaf():
            # Display leaf node with its label and error
            print(f"{prefix}{side} └── Leaf: Label = {node.label}")
        else:
            # Display split condition with a branch and indicate whether it's the left or right branch
            print(f"{prefix}{side} ├── Split on '{node.split_feature}' <= {node.split_value}")

            # Recursively display the left subtree with "Left branch" label
            left_prefix = prefix + "│   "
            self.draw_tree(node.left, depth + 1, left_prefix, "Left branch ")

            # Recursively display the right subtree with "Right branch" label
            right_prefix = prefix + "    "
            self.draw_tree(node.right, depth + 1, right_prefix, "Right branch")


def question_2():
    # File path to the dataset
    file_path = "iris.txt"

    # Step 1: Process the file and load the points
    data = process_file(file_path)
    all_p = type_arrays.get("all")

    # Step 2: Initialize parameters
    max_levels = 2  # k = 3, tree with root, children, and leaf grandchildren (levels 0,1,2)

    # Step 3: Create DecisionTree objects for brute force
    print("\n--- Brute Force Tree Construction (Strategy A) ---")
    brute_tree = DecisionTree(all_p, max_levels)
    best_tree, brute_error_count = brute_tree.brute_force()
    brute_error = brute_error_count / len(all_p)
    print(f"Brute force error count: {brute_error_count}")
    print("Brute Force Tree:")
    brute_tree.draw_tree(best_tree)

    # Step 4: Create DecisionTree objects for binary entropy
    k = max_levels  # 3-level tree
    print("\n--- Binary Entropy Tree Construction (Strategy B) --\n")
    entropy_tree = DecisionTree(all_p, k)
    best_entropy_tree = entropy_tree.binary_entropy()
    entropy_error_count = entropy_tree.calculate_tree_error(entropy_tree.root)
    print(f"Binary entropy error count: {entropy_error_count}")
    entropy_error = entropy_error_count / len(all_p)
    print("Binary Entropy Tree:")
    entropy_tree.draw_tree(best_entropy_tree)


    # Step 5: Compare errors
    print("\n--- Comparison ---")
    print(f"Error using BRUTE FORCE: {brute_error}")
    print(f"Error using binary ENTROPY: {entropy_error}")


if __name__ == "__main__":
    question_2()

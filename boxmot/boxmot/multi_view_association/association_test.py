import numpy as np
from scipy.optimize import linear_sum_assignment

def l2_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def match_objects(list1, list2, threshold):
    # Extract coordinates
    coords1 = [item[1] for item in list1]
    coords2 = [item[1] for item in list2]
    
    # Create cost matrix
    cost_matrix = np.zeros((len(coords1), len(coords2)))
    for i, c1 in enumerate(coords1):
        for j, c2 in enumerate(coords2):
            cost_matrix[i, j] = l2_distance(c1, c2)
    
    # Apply Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Prepare results
    matched_pairs = []
    used_indices = set()
    
    for i, j in zip(row_ind, col_ind):
        distance = cost_matrix[i, j]
        if distance <= threshold:
            matched_pairs.append((list1[i], list2[j]))
            used_indices.add(i)
            used_indices.add(j)
    
    # Unmatched items
    unmatched_list1 = [item for idx, item in enumerate(list1) if idx not in used_indices]
    unmatched_list2 = [item for idx, item in enumerate(list2) if idx not in used_indices]
    
    return matched_pairs, unmatched_list1, unmatched_list2

# Example usage
list1 = [("A", (1.0, 2.0)), ("B", (3.0, 4.0)), ("C", (10.0, 10.0))]
list2 = [("X", (1.1, 2.1)), ("Y", (3.1, 4.1)), ("Z", (20.0, 20.0))]
threshold = 0.5

matched_pairs, unmatched_list1, unmatched_list2 = match_objects(list1, list2, threshold)

# Output the results
output = []
for pair in matched_pairs:
    output.append(f"({pair[0][0]}, {pair[1][0]})")

for item in unmatched_list1:
    output.append(item[0])

for item in unmatched_list2:
    output.append(item[0])

print("Matched and unmatched objects:")
print(", ".join(output))
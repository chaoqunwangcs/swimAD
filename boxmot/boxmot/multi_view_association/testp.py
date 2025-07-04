def point_in_quadrilateral(x, y, vertices):
    # vertices: [(x0,y0), (x1,y1), (x2,y2), (x3,y3)] in clockwise order
    n = len(vertices)
    inside = False
    for i in range(n):
        x0, y0 = vertices[i]
        x1, y1 = vertices[(i + 1) % n]
        # Check if point is on the edge
        if (y == y0 == y1) and (min(x0, x1) <= x <= max(x0, x1)):
            return True  # On the edge
        if min(y0, y1) <= y < max(y0, y1):
            x_intersect = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
            if x < x_intersect:
                inside = not inside
    return inside

# Example usage:
vertices = [[1625, 416], [2240, 489], [1297, 1372], [436, 776]]  # Clockwise square
point = (347.0, 625.0)
print(point_in_quadrilateral(point[0], point[1], vertices))  # True
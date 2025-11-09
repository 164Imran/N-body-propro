from tkinter import Tk, Canvas
import numpy as np
import time
import random as rd

# Initialize the main window
root = Tk()
Side = 800  # Canvas size
cnv = Canvas(root, width=Side, height=Side)
cnv.pack()
cnv.config(bg="black")  # Set the background color to black
fps_label = cnv.create_text(60, 20, fill="white", text="")

# Variables for FPS calculation
last_time = time.perf_counter()
frames = 0
fps = 0

def square_list(n):
    """Creates a grid for spatial partitioning"""
    n0 = int(int(2**(2*(1+n)))**(1/2))
    return [[[] for k in range(n0)] for i in range(n0)]

def sqr_coord(point, dim, n0):
    """Returns the grid coordinates [i, j] and the center of the grid cell containing the point."""
    n = 2**(2*(n0+1))
    i, j = 0, 0

    while i*dim/(n**(1/2)) <= point[0][0][0]:
        i = i+1

    while j*dim/(n**(1/2)) <= point[0][0][1]:
        j = j+1

    return [[i, j], [((j+1)*dim/n + (j+2)*dim/n) / 2, ((i+1)*dim/n + i*dim/n)/2]]

def point_coord(points, n):
    """Returns the grid of points placed in the grid based on their positions"""
    global Side
    Side0 = Side * 10 ** 6
    l = square_list(n)
    for k in points:
        o = sqr_coord(k, Side0, n)
        l[o[0][0]][o[0][1]].append([o[1], k])
    return l

class Equiv:
    def __init__(self, items):
        u_x, u_y, item, dx, dy, m = items
        self.u_x = u_x  # x position
        self.u_y = u_y  # y position
        self.dx = dx    # velocity in x direction
        self.dy = dy    # velocity in y direction
        self.m = m      # mass
        self.item = item  # reference to the visual item (circle)

    def phykics(self):
        """Handles bouncing and physics against the walls of the simulation"""
        if self.u_x - r <= 0:  # Left wall collision
            self.dx = -self.dx * coef
            self.u_x = r
        if self.u_x + r >= Side:  # Right wall collision
            self.dx = -self.dx * coef
            self.u_x = Side - r
        if self.u_y - r <= 0:  # Top wall collision
            self.dy = -self.dy * coef
            self.u_y = r
        if self.u_y + r >= Side:  # Bottom wall collision
            self.dy = -self.dy * coef
            self.u_y = Side - r

    def update_canvas_position(self):
        """Updates the position of the object on the canvas"""
        x0 = self.u_x - r
        x1 = self.u_x + r
        y0 = self.u_y - r
        y1 = self.u_y + r
        cnv.coords(self.item, x0, y0, x1, y1)

l = []
d2 = np.sqrt(2)

# Parameters for the simulation
m1, m2, m3, m4 = 10**14, 10**14, 10**14, 10**1
x, y = 250, 350
x1, y1 = 250, 15

G = 6.67*10**(-11)  # Gravitational constant
coef = .5  # Coefficient for the collision response
dt = .0001  # Time step for the simulation
r = 1  # Radius of the particles

def update_canvas_position(x, y, item):
    """Updates the position of an object (oval) on the canvas"""
    x0 = x - r
    x1 = x + r
    y0 = y - r
    y1 = y + r
    cnv.coords(item, x0, y0, x1, y1)

def coord(item):
    """Returns the center coordinates of the given item (oval)"""
    return (cnv.coords(item)[0] + cnv.coords(item)[2]) / 2, (cnv.coords(item)[1] + cnv.coords(item)[3]) / 2

def f(r, m_1, m_2):
    """Calculates the gravitational force between two masses"""
    return G * m_1 * m_2 / (r ** 2)

def oval(x, y, r, color):
    """Creates an oval (representing a body) on the canvas"""
    x0 = x - r
    x1 = x + r
    y0 = y - r
    y1 = y + r
    return cnv.create_oval(x0, y0, x1, y1, fill=color)

def compute_force(POS, M):
    """Calculates the gravitational force between all bodies"""
    D = POS[:, None, :] - POS[None, :, :]  # Vector between each pair of particles
    DIST = np.linalg.norm(D, axis=2) * 10**6 + 1e-6  # Avoid division by zero
    F = - G * M * M.T / DIST ** 3  # Gravitational force calculation
    F = F[..., None] * D  # Apply direction to the force vectors
    return np.sum(F, axis=1)

def init_conditions(n):
    """Initialize the conditions of the simulation (randomly place bodies with random velocities)"""
    global r

    # Initialize the central body (massive object at the center)
    POS, VIT, M, ITEM = np.array([[Side//2, Side//2]]), np.array([[0, 0]]), np.array([[10**38]]), [oval(400, 400, r, 'red')]

    for _ in range(n):
        theta = rd.uniform(0, 2 * np.pi)  # Random angle around the center
        r_min = 80  # Minimum distance from the center (to avoid particles being too close to the center)
        r_max = 400  # Maximum radius for particles' positions
        r_dist = rd.uniform(r_min, r_max)  # Random distance between r_min and r_max
        x = r_dist * np.cos(theta) + Side / 2  # Convert to Cartesian coordinates
        y = r_dist * np.sin(theta) + Side / 2

        M_total = np.sum(M)  # Total mass (central mass in this case)
        v = np.sqrt(G * M_total / r_dist) / 10**9  # Scaling factor for velocity
        vx = -v * np.sin(theta) / 1000  # Orbital velocity in the x direction
        vy = v * np.cos(theta) / 1000  # Orbital velocity in the y direction

        item = oval(x, y, r, "white")
        m = np.random.uniform(10**10, 10**13)  # Random mass for the particle

        # Add new particle data
        POS = np.vstack((POS, [x, y]))
        M = np.vstack((M, [m]))
        VIT = np.vstack((VIT, [vx, vy]))
        ITEM.append(item)

    POS = np.array(POS, dtype=float)  # Shape (N, 2)
    VIT = np.array(VIT, dtype=float)  # Shape (N, 2)
    M = np.array(M, dtype=float)

    return POS, VIT, M, ITEM

i0 = 0
def anim_rebond(items):
    """Main animation loop to update positions and handle physics"""
    global r, last_time, frames, fps, i0
    i0 = i0 + 1
    start = time.perf_counter()

    POS, VIT, M, ITEM = items
    A = compute_force(POS, M) / M  # Gravitational accelerations
    VIT += A * dt  # Update velocities
    POS += VIT * dt  # Update positions

    # Handle collisions with the walls
    mask_x_left = POS[:, 0] - r <= 0
    VIT[mask_x_left, 0] *= -coef
    POS[mask_x_left, 0] = r

    mask_x_right = POS[:, 0] - r >= Side
    VIT[mask_x_right, 0] *= -coef
    POS[mask_x_right, 0] = Side - r

    mask_y_left = POS[:, 1] - r <= 0
    VIT[mask_y_left, 1] *= -coef
    POS[mask_y_left, 1] = r

    mask_y_right = POS[:, 1] - r >= Side
    VIT[mask_y_right, 1] *= -coef
    POS[mask_y_right, 1] = Side - r

    items = POS, VIT, M, ITEM
    for i in range(len(POS)):
        update_canvas_position(POS[i][0], POS[i][1], ITEM[i])

    frames += 1
    elapsed = start - last_time
    if elapsed >= 1.0:
        fps = frames / elapsed
        frames = 0
        last_time = start
        cnv.itemconfig(fps_label, text=f"{fps:.1f} FPS")  # Display FPS
    cnv.after(10, anim_rebond, items)  # Repeat the animation loop

# Example usage
n = 50  # Number of bodies, up to 600 particules, the animation can't exeed 10 fps
items = init_conditions(n)
anim_rebond(items)  # Start the animation
root.mainloop()  # Run the Tkinter main loop

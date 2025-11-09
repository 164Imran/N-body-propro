from tkinter import Tk, Canvas
import numpy as np

# Initialize the main window
root = Tk()
Side = 800  # Canvas size
cnv = Canvas(root, width=Side, height=Side)
cnv.pack()
cnv.config(bg="black")  # Set background to black

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
        # Function that handles bouncing against the boundaries of the simulation (physics)

        # Left wall
        if self.u_x - r <= 0:
            self.dx = -self.dx * coef
            self.u_x = r
            print(1, self.u_x, 451485)

        # Right wall
        if self.u_x + r >= Side:
            self.dx = -self.dx * coef
            self.u_x = Side - r

        # Top wall
        if self.u_y - r <= 0:
            self.dy = -self.dy * coef
            self.u_y = r

        # Bottom wall
        if self.u_y + r >= Side:
            self.dy = -self.dy * coef
            self.u_y = Side - r

    def update_canvas_position(self):
        """Update the object's position on the canvas"""
        x0 = self.u_x - r
        x1 = self.u_x + r
        y0 = self.u_y - r
        y1 = self.u_y + r
        cnv.coords(self.item, x0, y0, x1, y1)

l = []
d2 = np.sqrt(2)

# Constants for physics simulation

G = 6.67*10**(-11)  # Gravitational constant
coef = .3  # Coefficient of restitution for bouncing
dt = .0001  # Time step for simulation
r = 3  # Radius of the bodies

def coord(item):
    """Returns the center coordinates of the given item (oval)"""
    return (cnv.coords(item)[0] + cnv.coords(item)[2]) / 2, (cnv.coords(item)[1] + cnv.coords(item)[3]) / 2

def f(r,m_1, m_2):
    """Gravitational force calculation between two masses"""
    return G * m_1*m_2/(r**2)

def compute_force(body1, body2):
    """Compute the gravitational force between two bodies"""

    x = body2.u_x - body1.u_x  # Difference in x coordinates
    y = body2.u_y - body1.u_y  # Difference in y coordinates

    dist = np.sqrt(x ** 2 + y ** 2) * 10 ** 6  # Distance between the bodies in meters

    if dist < 1:  # Prevent division by zero in case of very small distance
        return np.array([0.0, 0.0])

    F = f(dist, body1.m, body2.m)  # Force calculation using Newton's law of gravitation
    direction = np.array([x/dist, y/dist])  # Direction of the force

    return F * direction  # Return the force vector

def anim_rebond(items):
    """Main animation loop, simulating bouncing and interactions between bodies"""
    global r
    for i in range(len(items)):
        F = np.array([0, 0])  # Reset the force for this body
        Planet = Equiv(items[i])

        for j in range(len(items)):  # Check for forces from all other bodies
            if j != i:
                Planet1 = Equiv(items[j])
                F = F + compute_force(Planet, Planet1)  # Sum the forces from other bodies

        ax, ay = F / Planet.m  # Calculate acceleration based on Newton's second law
        Planet.dx = Planet.dx + ax * dt  # Update velocity in the x direction
        Planet.dy = Planet.dy + ay * dt  # Update velocity in the y direction

        Planet.u_x += Planet.dx  # Update position based on velocity
        Planet.u_y += Planet.dy
        Planet.phykics()  # Handle bouncing off the walls

        items[i] = (Planet.u_x, Planet.u_y, Planet.item, Planet.dx, Planet.dy, Planet.m)

        Planet.update_canvas_position()  # Update the visual position on the canvas

    # Schedule the next frame of the animation
    cnv.after(int(dt * 10000), anim_rebond, items)

def oval(x, y, r, color):
    """Function to create an oval on the canvas (used to represent bodies)"""
    x0 = x - r
    x1 = x + r
    y0 = y - r
    y1 = y + r
    return cnv.create_oval(x0, y0, x1, y1, fill=color)

def init_conditions(n):
    """Initialize the conditions of the simulation (randomly place bodies)"""
    global r
    items = []
    for _ in range(n):
        m = np.random.uniform(10**10, 10**13)  # Random mass
        x = np.random.uniform(50, 800)  # Random x position
        y = np.random.uniform(50, 800)  # Random y position
        dx = np.random.uniform(-2.5, 2.5)  # Random velocity in x direction
        dy = np.random.uniform(-2.5, 2.5)  # Random velocity in y direction
        item = oval(x, y, r, "white")  # Create a body (oval)
        items.append((x, y, item, dx, dy, m))
    items.append((Side//2, Side//2, oval(400, 400, r, 'red'), 1, 0, 10**35))  # Add a large central body
    return items

# Example usage
n = 250  # Number of bodies (simulation starts slowing down after about 100 bodies)
items = init_conditions(n)
anim_rebond(items)
root.mainloop()

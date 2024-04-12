import numpy as np
from vispy import app, scene
from vispy.geometry import Rect
from funcs import init_boids, directions, propagate, flocking
from vispy.scene.visuals import Text


import pyglet
app.use_app('pyglet')

w, h = 1920, 1080
N = 5000
dt = 0.01
asp = w / h
perception = 1/20
vrange=(0, 0.1)
perception_angle = np.pi


#                    c      a    s    w   ns
coeffs = np.array([0.01, 0.05, 0.1, 0.2, 0.03])
# 0  1   2   3   4   5
# x, y, vx, vy, ax, ay
boids = np.zeros((N, 6), dtype=np.float64)
init_boids(boids, asp, vrange=vrange)
boids[:, 4:6] = 0.1

canvas = scene.SceneCanvas(show=True, size=(w, h))
view = canvas.central_widget.add_view()
view.camera = scene.PanZoomCamera(rect=Rect(0, 0, asp, 1))

arrows = scene.Arrow(arrows=directions(boids, dt),
                     arrow_color=[(0, 0, 1, 1)] + [(1, 1, 1, 1)] * (N - 1),
                     arrow_size=10,
                     connect='segments',
                     parent=view.scene)

txt = Text(parent=canvas.scene, color='white', bold=True, font_size= 12, pos=(0.5*w, 0.05*h))




def update(event):
    flocking(boids, perception, perception_angle, coeffs, asp, vrange)
    propagate(boids, dt, vrange)
    arrows.set_data(arrows=directions(boids, dt))

    agent_count = boids.shape[0]
    params_info = f"Cohesion: {coeffs[0]}, Alignment: {coeffs[1]}, Separation: {coeffs[2]}, Walls: {coeffs[3]}, Noise: {coeffs[4]}"
    fps_info = f"FPS: {canvas.fps:.2f}"

    txt.text = f"Agents: {agent_count}\nParameters: {params_info}\n{fps_info}"
    canvas.update()


if __name__ == '__main__':
    timer = app.Timer(interval=0, start=True, connect=update)
    canvas.measure_fps()
    app.run()

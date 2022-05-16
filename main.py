import matplotlib.pyplot as plt
import numpy as np
import time


MEASUREMENT_SIGMA_SQUARE = 15.0  # measurement error dispersion
GROUND_ANGLE_ERROR = 5.0 / 180.0 * np.pi  # car azimuth moving error (sigma)
DISTANCE_ERROR = 20.0  # car distance moving error (sigma)
NUM_PARTICLES = 1000
NUM_ITERATIONS = 50
MAX_X = 100.0
MAX_Y = 100.0
LANDMARK_MAX_X = 1000
LANDMARK_MAX_Y = 1000

# car controls on trajectory (spiral)
CONTROL_DISTANCES = np.linspace(50, 70, NUM_ITERATIONS)
CONTROL_DISTANCES[0] = 0
CONTROL_ANGLES = np.zeros(NUM_ITERATIONS) + 10.0 / 180.0 * np.pi


class Landmark:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Car:
    def __init__(self, x, y, azimuth):
        self.x = x
        self.y = y
        self.azimuth = azimuth

    def move(self, distance, ground_angle):
        self.azimuth += np.random.normal(ground_angle, GROUND_ANGLE_ERROR)
        random_distance = np.random.normal(distance, DISTANCE_ERROR)
        self.x += random_distance * np.cos(self.azimuth)
        self.y += random_distance * np.sin(self.azimuth)

    def measure(self, landmarks):
        measurements = [np.sqrt((self.x - landmark.x)**2 + (self.y - landmark.y)**2) for landmark in landmarks]
        return measurements

    def coords(self):
        return self.x, self.y, self.azimuth


class Particle:
    def __init__(self, x, y, azimuth, weight):
        self.x = x
        self.y = y
        self.azimuth = azimuth
        self.weight = weight

    def move(self, distance, ground_angle):
        self.azimuth += np.random.normal(ground_angle, GROUND_ANGLE_ERROR)
        random_distance = np.random.normal(distance, DISTANCE_ERROR)
        self.x += random_distance * np.cos(self.azimuth)
        self.y += random_distance * np.sin(self.azimuth)

    def measure(self, landmarks, base_measurements):
        self.weight = 1.0
        for landmark, car_measurement in zip(landmarks, base_measurements):
            measurement = np.sqrt((self.x - landmark.x)**2 + (self.y - landmark.y)**2)
            probability = 1.0 / np.sqrt(2 * np.pi * MEASUREMENT_SIGMA_SQUARE) * \
                          np.exp(-(measurement - car_measurement)**2 / MEASUREMENT_SIGMA_SQUARE)
            self.weight *= probability


def norm_weights(particles):
    sum_weight = np.sum([p.weight for p in particles])
    print('sum_weight', sum_weight)
    for i, p in enumerate(particles):
        particles[i].weight = p.weight / sum_weight
    return particles


def resampling(particles):
    new_particles_list = []
    N = len(particles)
    index = np.random.randint(0, N)
    betta = 0.0
    weights = [p.weight for p in particles]
    max_weight = max(weights)
    print('max_weight', max_weight)
    for i in range(N):
        betta += np.random.uniform(0, 2 * max_weight)
        while betta > particles[index].weight:
            betta -= particles[index].weight
            index = (index + 1) % N
        new_particles_list.append(Particle(particles[index].x,
                                           particles[index].y,
                                           particles[index].azimuth,
                                           particles[index].weight))
    return new_particles_list


def estimate(particles):
    x_est = 0.0
    y_est = 0.0
    azimuth_estimate = 0.0
    for p in particles:
        x_est += p.x * p.weight
        y_est += p.y * p.weight
        azimuth_estimate += p.azimuth * p.weight
    return x_est, y_est, azimuth_estimate


if __name__ == '__main__':
    landmarks_map = [Landmark(LANDMARK_MAX_X, LANDMARK_MAX_Y),
                     Landmark(LANDMARK_MAX_X, -LANDMARK_MAX_Y),
                     Landmark(-LANDMARK_MAX_X, -LANDMARK_MAX_Y),
                     Landmark(-LANDMARK_MAX_X, LANDMARK_MAX_Y)]
    particles_list = [Particle(np.random.uniform(-MAX_X, MAX_Y),
                               np.random.uniform(-MAX_X, MAX_Y),
                               np.random.uniform(-np.pi, np.pi),
                               1.0 / NUM_PARTICLES) for i in range(NUM_PARTICLES)]
    car = Car(0.0, 0.0, 0.0)
    position_estimate = []
    position_true = []
    print('Init ready!')
    x_particles = [p.x for p in particles_list]
    y_particles = [p.y for p in particles_list]
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    line1, line2, = ax.plot(x_particles, y_particles, 'b-', car.x, car.y, 'ro')
    plt.xlim([-LANDMARK_MAX_X, LANDMARK_MAX_X])
    plt.ylim([-LANDMARK_MAX_X, LANDMARK_MAX_X])
    plt.show()
    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(1)

    for i in range(NUM_ITERATIONS):
        car.move(CONTROL_DISTANCES[i], CONTROL_ANGLES[i])
        car_measurements = car.measure(landmarks_map)
        for p in particles_list:
            p.move(CONTROL_DISTANCES[i], CONTROL_ANGLES[i])
            p.measure(landmarks_map, car_measurements)
        particles_list = norm_weights(particles_list)
        if (i < 5) or (i % 5 == 0):
            particles_list = resampling(particles_list)
            particles_list = norm_weights(particles_list)
        position_estimate.append(estimate(particles_list))
        position_true.append(car.coords())
        x_particles = [p.x for p in particles_list]
        y_particles = [p.y for p in particles_list]
        line1.set_ydata(y_particles)
        line1.set_xdata(x_particles)
        fig.canvas.draw()
        fig.canvas.flush_events()
        line2.set_xdata(car.x)
        line2.set_ydata(car.y)
        fig.canvas.draw()
        fig.canvas.flush_events()

    x_true = [pos[0] for pos in position_true]
    y_true = [pos[1] for pos in position_true]
    x_estimate = [pos[0] for pos in position_estimate]
    y_estimate = [pos[1] for pos in position_estimate]
    plt.clf()
    plt.plot(x_true, y_true, 'r*', x_estimate, y_estimate, 'b*')
    plt.show()
    input()





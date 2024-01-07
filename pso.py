import copy
import numpy as np

def initialize_particles(n_particles,n_dims, hyperparameters):
    result = hyperparameters.values()
# Convert object to a list
    data = list(result)
# Convert list to an array
    numpyArray = np.array(data)
# Generate initial particles randomly
    particles = []
    for i in range(n_particles):
        particle = []
        for j in range(n_dims):
            particle.append(random.randrange(numpyArray[j][0], numpyArray[j][1]+1,numpyArray[j][2]))
        #particle.append(random.uniform(numpyArray[n_dims-1][0], numpyArray[n_dims-1][1]))
​
        particles.append(particle)
    return particles
​
def update_velocity(particle, velocity, best_particle, global_best, c1, c2, w):
    """Update the velocity of a particle based on its previous velocity, its personal best,
    and the global best."""
    for i in range(len(particle)):
        r1 = random.randrange(0,2)
        r2 = random.randrange(0,2)
    # if type(particle[i])==int:
        vel_cognitive = c1 * r1 * (best_particle[i] - particle[i])
        vel_social = c2 * r2 * (global_best[i] - particle[i])
        velocity[i] = velocity[i] + vel_cognitive + vel_social
          
        """vel_cognitive = c1 * r1 * (best_particle[i] - particle[i])
            vel_social = c2 * r2 * (global_best[i] - particle[i])
            velocity[i] = w*velocity[i] + vel_cognitive + vel_social"""
    return velocity
​
def update_particle(particle, velocity):
    """Update the position of a particle based on its velocity."""
    for i in range(len(particle)):
        particle[i] += velocity[i]
    return particle
​
def run_pso(n_particles, n_dims, hyperparameters, c1, c2, w, max_iter):
    """Run the PSO algorithm to optimize the hyperparameters of a CNN model."""
    particles = initialize_particles(n_particles,n_dims, hyperparameters)
    velocities = [[0 for j in range(n_dims)] for i in range(n_particles)]
    personal_best_accuracies = [0 for i in range(n_particles)]
    personal_best_positions = particles.copy()
    global_best = particles[0].copy()
    global_best_accuracy = evaluate_model(global_best)
    
    for t in range(max_iter):
        print("Iteration:",t)
        for i in range(n_particles):
            accuracy = evaluate_model(particles[i])
            if accuracy > personal_best_accuracies[i]:
                personal_best_accuracies[i] = accuracy
                personal_best_positions[i] = particles[i].copy()
                if accuracy > global_best_accuracy:
                    global_best_accuracy = accuracy
                    global_best = particles[i].copy()
        for i in range(n_particles):
            velocity = update_velocity(particles[i], velocities[i], personal_best_positions[i], global_best, c1,c2, w)
            particle = update_particle(particles[i], velocity)
            particles[i] = particle
​
    return global_best, global_best_accuracy


#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <raylib.h>
#include <cstdlib>
#include <ctime>
#include <cmath>

// Constants
#define MAX_PARTICLES 5000  // Maximum number of particles to simulate
#define BLOCK_SIZE 256      // Number of threads per block for CUDA kernels

/////// CPU Performance Notes ///////
// 2000 = 14 FPS                    //
// 1000 = 53 FPS                    //
// 500 = 142 FPS                    //
//////////////////////////////////////
/////// GPU First Version Notes //////
// 5000 = 40 FPS                    //
// 4000 = 50 FPS                    //
// 3000 = 66 FPS                    //
// 2000 = 96 FPS                    //
// 1000 = 142 FPS                   //
// 500 = 144 FPS                    //
//////////////////////////////////////

// Device constants for GPU interaction
// These are initialized on the host and copied to the GPU using cudaMemcpyToSymbol
__constant__ float d_MAX_DISTANCE;           // Maximum distance for particle interactions
__constant__ float d_MIN_DISTANCE;           // Minimum distance to avoid division by zero
__constant__ float d_FORCE_STRENGTH;         // Strength of attraction/repulsion forces
__constant__ float d_MIN_COLLISION_DISTANCE; // Distance threshold for collisions
__constant__ float d_MAX_SPEED;              // Maximum speed of particles
__constant__ float d_MIN_SPEED;              // Minimum speed of particles

// Particle structure
// Represents a single particle with position, velocity, and color
struct Particle {
    Vector2 position;  // 2D position of the particle
    Vector2 velocity;  // 2D velocity of the particle
    Color color;       // Particle color (RGBA)
};

// Caps the velocity of a particle between specified limits
// Ensures particles do not exceed physical speed constraints
__device__ void CapSpeed(Vector2& velocity, float maxSpeed, float minSpeed) {
    float speed = sqrtf(velocity.x * velocity.x + velocity.y * velocity.y);
    if (speed > maxSpeed) {
        velocity.x = (velocity.x / speed) * maxSpeed;
        velocity.y = (velocity.y / speed) * maxSpeed;
    }
    if (speed < minSpeed && speed > 0) {
        velocity.x = (velocity.x / speed) * minSpeed;
        velocity.y = (velocity.y / speed) * minSpeed;
    }
}

// CUDA kernel to update particle interactions
// Computes forces between particles, updates velocities, and adjusts positions
__global__ void UpdateParticleInteractions(Particle* particles, int particleCount, int screenWidth, int screenHeight) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Calculate the global thread index
    if (i >= particleCount) return;               // Exit if index is out of bounds

    Particle& p1 = particles[i];                  // Reference to the particle being updated
    for (int j = 0; j < particleCount; ++j) {     // Loop over all other particles
        if (i == j) continue;                     // Skip self-interaction

        Particle& p2 = particles[j];
        float dx = p2.position.x - p1.position.x; // Compute x-axis distance
        float dy = p2.position.y - p1.position.y; // Compute y-axis distance
        float distance = sqrtf(dx * dx + dy * dy); // Euclidean distance

        // Apply attraction/repulsion forces if within interaction range
        if (distance < d_MAX_DISTANCE && distance > d_MIN_DISTANCE) {
            float force = -d_FORCE_STRENGTH / distance; // Compute force magnitude
            Vector2 direction = { dx / distance, dy / distance }; // Normalize direction
            p1.velocity.x += direction.x * force;
            p1.velocity.y += direction.y * force;
        }

        // Apply collision response if particles are too close
        if (distance < d_MIN_COLLISION_DISTANCE) {
            Vector2 collisionDirection = { dx / distance, dy / distance };
            p1.velocity.x -= collisionDirection.x * d_FORCE_STRENGTH;
            p1.velocity.y -= collisionDirection.y * d_FORCE_STRENGTH;
        }
    }

    // Update particle position based on velocity
    p1.position.x += p1.velocity.x;
    p1.position.y += p1.velocity.y;

    // Ensure particles bounce off the screen edges
    if (p1.position.x >= screenWidth || p1.position.x <= 0) p1.velocity.x *= -1;
    if (p1.position.y >= screenHeight || p1.position.y <= 0) p1.velocity.y *= -1;

    CapSpeed(p1.velocity, d_MAX_SPEED, d_MIN_SPEED); // Limit velocity to allowed range
}


int main() {
    // Screen dimensions for rendering
    int screenWidth = 1440;
    int screenHeight = 920;
    InitWindow(screenWidth, screenHeight, "Particle Interaction - CUDA");

    srand(static_cast<unsigned int>(time(0))); // Seed for random number generation

    // Allocate and initialize particles on the host
    Particle* h_particles = new Particle[MAX_PARTICLES];
    for (int i = 0; i < MAX_PARTICLES; i++) {
        h_particles[i].position = { (float)(rand() % screenWidth), (float)(rand() % screenHeight) };
        h_particles[i].velocity = { (float)(rand() % 5 - 2), (float)(rand() % 5 - 2) };
        h_particles[i].color = { (unsigned char)(rand() % 256), (unsigned char)(rand() % 256), (unsigned char)(rand() % 256), 255 };
    }

    // Initialize interaction constants on the GPU
    float h_MAX_DISTANCE = 14.0f;
    float h_MIN_DISTANCE = 8.0f;
    float h_FORCE_STRENGTH = 5.0f;
    float h_MIN_COLLISION_DISTANCE = 10.0f;
    float h_MAX_SPEED = 2.5f;
    float h_MIN_SPEED = 0.1f;

    cudaMemcpyToSymbol(d_MAX_DISTANCE, &h_MAX_DISTANCE, sizeof(float));
    cudaMemcpyToSymbol(d_MIN_DISTANCE, &h_MIN_DISTANCE, sizeof(float));
    cudaMemcpyToSymbol(d_FORCE_STRENGTH, &h_FORCE_STRENGTH, sizeof(float));
    cudaMemcpyToSymbol(d_MIN_COLLISION_DISTANCE, &h_MIN_COLLISION_DISTANCE, sizeof(float));
    cudaMemcpyToSymbol(d_MAX_SPEED, &h_MAX_SPEED, sizeof(float));
    cudaMemcpyToSymbol(d_MIN_SPEED, &h_MIN_SPEED, sizeof(float));

    // Allocate device memory for particles and copy data from host
    Particle* d_particles;
    cudaMalloc(&d_particles, MAX_PARTICLES * sizeof(Particle));
    cudaMemcpy(d_particles, h_particles, MAX_PARTICLES * sizeof(Particle), cudaMemcpyHostToDevice);

    SetTargetFPS(144); // Set the frame rate for rendering

    // Main simulation loop
    while (!WindowShouldClose()) {
        // Launch the CUDA kernel for particle interaction updates
        int blocks = (MAX_PARTICLES + BLOCK_SIZE - 1) / BLOCK_SIZE;
        UpdateParticleInteractions << <blocks, BLOCK_SIZE >> > (d_particles, MAX_PARTICLES, screenWidth, screenHeight);
        cudaDeviceSynchronize(); // Ensure kernel execution is complete

        // Copy updated particle data back to the host for rendering
        cudaMemcpy(h_particles, d_particles, MAX_PARTICLES * sizeof(Particle), cudaMemcpyDeviceToHost);

        // Render particles using raylib
        BeginDrawing();
        ClearBackground(BLACK);
        for (int i = 0; i < MAX_PARTICLES; i++) {
            DrawCircleV(h_particles[i].position, 7.0f, h_particles[i].color);
        }
        DrawText(TextFormat("FPS: %i", GetFPS()), 10, 10, 20, WHITE); // Display FPS
        EndDrawing();
    }

    // Cleanup memory and close the window
    delete[] h_particles;
    cudaFree(d_particles);
    CloseWindow();

    return 0;
}
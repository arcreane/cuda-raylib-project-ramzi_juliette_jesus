
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <raylib.h>
#include <cstdlib>
#include <ctime>
#include <cmath>

// Constants
#define MAX_PARTICLES 500
#define BLOCK_SIZE 256

// 2000 = 14 FPS
// 1000 = 53 FPS
// 500 = 142 FPS
const float FORCE_STRENGTH = 5.0f;  // Attraction/repulsion force constant
const float MIN_DISTANCE = 8.0f;   // Minimum distance for interaction (avoid division by zero)
const float MAX_DISTANCE = 14.0f;  // Maximum distance for interaction (particles won't affect each other beyond this)
const float MAX_SPEED = 2.5f;      // Maximum speed for particles
const float MIN_SPEED = 0.1f;      // Minimum speed for particles
const float MIN_COLLISION_DISTANCE = 10.0f; // Minimum distance for particles to collide and bounce

// Particle struct definition
struct Particle {
    Vector2 position;
    Vector2 velocity;
    Color color;
};

// CUDA error check macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Cap speed of a particle
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

// Kernel to update particle interactions
__global__ void UpdateParticleInteractions(Particle* particles, int particleCount, int screenWidth, int screenHeight) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= particleCount) return;

    Particle& p1 = particles[i];
    for (int j = 0; j < particleCount; ++j) {
        if (i == j) continue;

        Particle& p2 = particles[j];
        float dx = p2.position.x - p1.position.x;
        float dy = p2.position.y - p1.position.y;
        float distance = sqrtf(dx * dx + dy * dy);

        if (distance < MAX_DISTANCE && distance > MIN_DISTANCE) {
            float force = -FORCE_STRENGTH / distance;
            Vector2 direction = { dx / distance, dy / distance };
            p1.velocity.x += direction.x * force;
            p1.velocity.y += direction.y * force;
        }

        if (distance < MIN_COLLISION_DISTANCE) {
            Vector2 collisionDirection = { dx / distance, dy / distance };
            p1.velocity.x -= collisionDirection.x * FORCE_STRENGTH;
            p1.velocity.y -= collisionDirection.y * FORCE_STRENGTH;
        }
    }

    // Update position
    p1.position.x += p1.velocity.x;
    p1.position.y += p1.velocity.y;

    // Bounce off screen edges
    if (p1.position.x >= screenWidth || p1.position.x <= 0) p1.velocity.x *= -1;
    if (p1.position.y >= screenHeight || p1.position.y <= 0) p1.velocity.y *= -1;

    CapSpeed(p1.velocity, MAX_SPEED, MIN_SPEED);
}

int main() {
    int screenWidth = 1440;
    int screenHeight = 920;
    InitWindow(screenWidth, screenHeight, "Particle Interaction - CUDA");

    srand(static_cast<unsigned int>(time(0)));

    // Host particles
    Particle* h_particles = new Particle[MAX_PARTICLES];

    // Initialize host particles
    for (int i = 0; i < MAX_PARTICLES; i++) {
        h_particles[i].position = { (float)(rand() % screenWidth), (float)(rand() % screenHeight) };
        h_particles[i].velocity = { (float)(rand() % 5 - 2), (float)(rand() % 5 - 2) };
        h_particles[i].color = { (unsigned char)(rand() % 256), (unsigned char)(rand() % 256), (unsigned char)(rand() % 256), 255 };
    }

    // Device particles
    Particle* d_particles;
    cudaMalloc(&d_particles, MAX_PARTICLES * sizeof(Particle));
    cudaMemcpy(d_particles, h_particles, MAX_PARTICLES * sizeof(Particle), cudaMemcpyHostToDevice);

    SetTargetFPS(144);

    while (!WindowShouldClose()) {
        // Launch kernel
        int blocks = (MAX_PARTICLES + BLOCK_SIZE - 1) / BLOCK_SIZE;
        UpdateParticleInteractions << <blocks, BLOCK_SIZE >> > (d_particles, MAX_PARTICLES, screenWidth, screenHeight);
        cudaDeviceSynchronize();

        // Copy updated particles back to host
        cudaMemcpy(h_particles, d_particles, MAX_PARTICLES * sizeof(Particle), cudaMemcpyDeviceToHost);

        BeginDrawing();
        ClearBackground(BLACK);

        for (int i = 0; i < MAX_PARTICLES; i++) {
            DrawCircleV(h_particles[i].position, 7.0f, h_particles[i].color);
        }

        DrawText(TextFormat("FPS: %i", GetFPS()), 10, 10, 20, WHITE);
        EndDrawing();
    }

    // Cleanup
    delete[] h_particles;
    cudaFree(d_particles);
    CloseWindow();

    return 0;
}
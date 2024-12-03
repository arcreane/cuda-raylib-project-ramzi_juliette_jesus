
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <raylib.h>
#include <cstdlib>
#include <ctime>
#include <cmath>

// Constants
#define MAX_PARTICLES 5000
#define BLOCK_SIZE 256

/////// CPU //////////////////////
// 2000 = 14 FPS                //
// 1000 = 53 FPS                //
// 500 = 142 FPS                //
//////////////////////////////////
/////// GPU first version ////////
// 5000 = 40 FPS                //
// 4000 = 50 FPS                //
// 3000 = 66 FPS                //
// 2000 = 96 FPS                //
// 1000 = 142 FPS               //
// 500 = 144 FPS                //
//////////////////////////////////
// 
__constant__ float d_MAX_DISTANCE;
__constant__ float d_MIN_DISTANCE;
__constant__ float d_FORCE_STRENGTH;
__constant__ float d_MIN_COLLISION_DISTANCE;
__constant__ float d_MAX_SPEED;
__constant__ float d_MIN_SPEED;

// Particle struct definition
struct Particle {
    Vector2 position;
    Vector2 velocity;
    Color color;
};

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

        if (distance < d_MAX_DISTANCE && distance > d_MIN_DISTANCE) {
            float force = -d_FORCE_STRENGTH / distance;
            Vector2 direction = { dx / distance, dy / distance };
            p1.velocity.x += direction.x * force;
            p1.velocity.y += direction.y * force;
        }

        if (distance < d_MIN_COLLISION_DISTANCE) {
            Vector2 collisionDirection = { dx / distance, dy / distance };
            p1.velocity.x -= collisionDirection.x * d_FORCE_STRENGTH;
            p1.velocity.y -= collisionDirection.y * d_FORCE_STRENGTH;
        }
    }

    // Update position
    p1.position.x += p1.velocity.x;
    p1.position.y += p1.velocity.y;

    // Bounce off screen edges
    if (p1.position.x >= screenWidth || p1.position.x <= 0) p1.velocity.x *= -1;
    if (p1.position.y >= screenHeight || p1.position.y <= 0) p1.velocity.y *= -1;

    CapSpeed(p1.velocity, d_MAX_SPEED, d_MIN_SPEED);
}


int main() {
    // Screen dimensions
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

    // Initialize constants on the device
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
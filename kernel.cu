#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <raylib.h>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <vector>

// Constants in device memory
__constant__ int d_screenWidth;
__constant__ int d_screenHeight;
__constant__ float d_FORCE_STRENGTH;
__constant__ float d_MIN_DISTANCE;
__constant__ float d_MAX_DISTANCE;
__constant__ float d_MAX_SPEED;
__constant__ float d_MIN_SPEED;
__constant__ float d_MIN_COLLISION_DISTANCE;
__constant__ float d_radius;

// Constants for display and shape
int h_screenWidth = 1920;
int h_screenHeight = 920;
float h_radius = 7.0f;

// Constants for interaction and collision
float h_FORCE_STRENGTH = 5.0f; // Attraction/repulsion force constant 
float h_MIN_DISTANCE = 2 * h_radius* 2 * h_radius; // Minimum distance for interaction (avoid division by zero)
float h_MAX_DISTANCE = 2.8 * h_radius* 2.8 * h_radius; // Maximum distance for interaction (particles won't affect each other beyond this)
float h_MAX_SPEED = 2.5f; // Maximum speed for particles
float h_MIN_SPEED = 0.1f; // Minimum speed for particles
float h_MIN_COLLISION_DISTANCE = 2.5 * h_radius * 2.5 * h_radius; // Minimum distance for particles to collide and bounce

// Flag to pause
bool pause = 0;

#define MAX_PARTICLES 2000
#define BLOCK_SIZE 256

/////// CPU Performance Notes //////////
// 2000 = 14 FPS                      //
// 1000 = 53 FPS                      //
// 500 = 142 FPS                      //
////////////////////////////////////////
/////// GPU First Version 1 Notes //////
// 5000 = 40 FPS                      //
// 4000 = 50 FPS                      //
// 3000 = 66 FPS                      //
// 2000 = 96 FPS                      //
// 1000 = 142 FPS                     //
// 500 = 144 FPS                      //
////////////////////////////////////////

// Particle struct definition
struct Particle {
    Vector2 position;
    Vector2 velocity;
    Color color;
};

// CUDA Kernel to limit the velocity of a particle to the maximum and minimum speeds
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

// CUDA Kernel to update the position of the particles and control the collision with the walls
__global__ void UpdateParticlesKernel(Particle* particles, int particleCount) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= particleCount) return;

    Particle& p = particles[i];
    p.position.x += p.velocity.x;
    p.position.y += p.velocity.y;

    // Cap speed
    CapSpeed(p.velocity, d_MAX_SPEED, d_MIN_SPEED);

    // Bounce off edges
    if (p.position.x >= d_screenWidth - d_radius || p.position.x <= d_radius) {
        p.velocity.x *= -1.0f;
        p.position.x = fminf(fmaxf(p.position.x, d_radius), d_screenWidth - d_radius);
    }
    if (p.position.y >= d_screenHeight - d_radius || p.position.y <= d_radius) {
        p.velocity.y *= -1.0f;
        p.position.y = fminf(fmaxf(p.position.y, d_radius), d_screenHeight - d_radius);
    }
}

// CUDA kernel to handle particle interactions
__global__ void HandleInteractionsKernel(Particle* particles, int particleCount) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= particleCount) return;

    Particle& p1 = particles[i];
    for (int j = 0; j < particleCount; ++j) {
        if (i == j) continue;

        Particle& p2 = particles[j];
        float dx = p2.position.x - p1.position.x;
        float dy = p2.position.y - p1.position.y;
        float distance = dx * dx + dy * dy;

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
}

// Function to initialize the particles with random values
void InitializeParticles(std::vector<Particle>& particles) {
    for (Particle& particle : particles) {
        particle.position = { (float)(rand() % h_screenWidth), (float)(rand() % h_screenHeight) };
        particle.velocity = { (float)(rand() % 5 - 2), (float)(rand() % 5 - 2) };
        particle.color = Color{ (unsigned char)(rand() % 256), (unsigned char)(rand() % 256), (unsigned char)(rand() % 256), 255 };
    }
}

// CUDA Kernel to check the Keyboard inputs and define the actions to take
__global__ void CheckKeyBoardInputKernel(Particle* particles, int particleCount, bool keyDown, bool keyUp, bool keyLeft, bool keyRight, bool keySpace, bool* pause, float maxSpeed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    Particle& p = particles[i];
    if (keyDown) {
        p.velocity = { 0.0f, maxSpeed };
    }
    if (keyUp) {
        p.velocity = { 0.0f, -maxSpeed };
    }
    if (keyLeft) {
        p.velocity = { -maxSpeed, 0.0f };
    }
    if (keyRight) {
        p.velocity = { maxSpeed, 0.0f };
    }
    if (keySpace) {
        *pause = !(*pause);
    }
}



int main() {

    // Set up window
    InitWindow(h_screenWidth, h_screenHeight, "Particle Interaction - GPU");

    srand(static_cast<unsigned int>(time(0)));

    // Host particles
    std::vector<Particle> h_particles(MAX_PARTICLES);
    InitializeParticles(h_particles);

    // Device particles
    Particle* d_particles;
    cudaMalloc(&d_particles, MAX_PARTICLES * sizeof(Particle));
    cudaMemcpy(d_particles, h_particles.data(), MAX_PARTICLES * sizeof(Particle), cudaMemcpyHostToDevice);

    // Pause state
    bool h_pause = false;  // Host pause state
    bool* d_pause;         // Device pause state
    cudaMalloc(&d_pause, sizeof(bool));
    cudaMemcpy(d_pause, &h_pause, sizeof(bool), cudaMemcpyHostToDevice);

    // Copy constants to device
    cudaMemcpyToSymbol(d_screenWidth, &h_screenWidth, sizeof(int));
    cudaMemcpyToSymbol(d_screenHeight, &h_screenHeight, sizeof(int));
    cudaMemcpyToSymbol(d_FORCE_STRENGTH, &h_FORCE_STRENGTH, sizeof(float));
    cudaMemcpyToSymbol(d_MIN_DISTANCE, &h_MIN_DISTANCE, sizeof(float));
    cudaMemcpyToSymbol(d_MAX_DISTANCE, &h_MAX_DISTANCE, sizeof(float));
    cudaMemcpyToSymbol(d_MAX_SPEED, &h_MAX_SPEED, sizeof(float));
    cudaMemcpyToSymbol(d_MIN_SPEED, &h_MIN_SPEED, sizeof(float));
    cudaMemcpyToSymbol(d_MIN_COLLISION_DISTANCE, &h_MIN_COLLISION_DISTANCE, sizeof(float));
    cudaMemcpyToSymbol(d_radius, &h_radius, sizeof(float));

    SetTargetFPS(144);

    while (!WindowShouldClose()) {

        // Gather keyboard input
        bool keyDown = IsKeyDown(KEY_DOWN);
        bool keyUp = IsKeyDown(KEY_UP);
        bool keyLeft = IsKeyDown(KEY_LEFT);
        bool keyRight = IsKeyDown(KEY_RIGHT);
        bool keySpace = IsKeyPressed(KEY_SPACE);  // Check if space is pressed

        // Update the pause state if SPACE is pressed
        if (keySpace) {
            h_pause = !h_pause;  // Toggle pause state on host
            cudaMemcpy(d_pause, &h_pause, sizeof(bool), cudaMemcpyHostToDevice);  // Sync pause state to device
        }

        if (!h_pause) {
            int blocks = (MAX_PARTICLES + BLOCK_SIZE - 1) / BLOCK_SIZE;

            // Update particles on GPU
            UpdateParticlesKernel << <blocks, BLOCK_SIZE >> > (d_particles, MAX_PARTICLES);
            cudaDeviceSynchronize();

            // Handle interactions on GPU
            HandleInteractionsKernel << <blocks, BLOCK_SIZE >> > (d_particles, MAX_PARTICLES);
            cudaDeviceSynchronize();

            // Update velocities based on keyboard input
            CheckKeyBoardInputKernel << <blocks, BLOCK_SIZE >> > (d_particles, MAX_PARTICLES, keyDown, keyUp, keyLeft, keyRight, keySpace, d_pause, h_MAX_SPEED);
            cudaDeviceSynchronize();


            // Copy updated particles back to host
            cudaMemcpy(h_particles.data(), d_particles, MAX_PARTICLES * sizeof(Particle), cudaMemcpyDeviceToHost);
        }

        BeginDrawing();
        ClearBackground(BLACK);
        for (const Particle& particle : h_particles) {
            DrawCircleV(particle.position, h_radius, particle.color);
        }
        DrawText(TextFormat("FPS: %i", GetFPS()), 10, 10, 20, WHITE);
        EndDrawing();
    }

    cudaFree(d_particles);
    cudaFree(d_pause);
    CloseWindow();

    return 0;
}

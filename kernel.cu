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

bool pause = false;
#define MAX_PARTICLES 2000
#define BLOCK_SIZE 256

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

// CUDA kernel to update particle positions
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
}

void InitializeParticles(std::vector<Particle>& particles) {
    for (Particle& particle : particles) {
        particle.position = { (float)(rand() % 1440), (float)(rand() % 920) };
        particle.velocity = { (float)(rand() % 5 - 2), (float)(rand() % 5 - 2) };
        particle.color = Color{ (unsigned char)(rand() % 256), (unsigned char)(rand() % 256), (unsigned char)(rand() % 256), 255 };
    }
}

__global__ void CheckKeyBoardInputKernel(Particle* particles, int particleCount, bool keyDown, bool keyUp, bool keyLeft, bool keyRight, bool keySpace, bool* pause, float maxSpeed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Toggle pause state if space is pressed
    //if (i == 0 && keySpace) {
    //    *pause = !(*pause);
    //}

    //if (*pause) return;  // Skip updates if paused

    //if (i >= particleCount) return;

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
    InitWindow(1440, 920, "Particle Interaction - GPU");

    srand(static_cast<unsigned int>(time(0)));

    // Host particles
    std::vector<Particle> h_particles(MAX_PARTICLES);
    InitializeParticles(h_particles);

    // Device particles
    Particle* d_particles;
    cudaMalloc(&d_particles, MAX_PARTICLES * sizeof(Particle));
    cudaMemcpy(d_particles, h_particles.data(), MAX_PARTICLES * sizeof(Particle), cudaMemcpyHostToDevice);

    // Host-side constants
    int h_screenWidth = 1440;
    int h_screenHeight = 920;
    float h_radius = 7.0f;
    float h_FORCE_STRENGTH = 5.0f;
    float h_MIN_DISTANCE = 2 * h_radius;
    float h_MAX_DISTANCE = 3.2* h_radius;
    float h_MAX_SPEED = 2.5f;
    float h_MIN_SPEED = 0.1f;
    float h_MIN_COLLISION_DISTANCE = 2.5 * h_radius;
    

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

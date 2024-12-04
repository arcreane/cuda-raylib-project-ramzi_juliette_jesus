#include <raylib.h>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <vector>

using namespace std;

// Constants for controlling the number of particles and interaction forces
#define MAX_PARTICLES 1000
// 2000 = 14 FPS
// 1000 = 53 FPS
// 500 = 142 FPS
const int screenWidth = 1440;
const int screenHeight = 920;
const float FORCE_STRENGTH = 5.0f;  // Attraction/repulsion force constant
const float MIN_DISTANCE = 8.0f;   // Minimum distance for interaction (avoid division by zero)
const float MAX_DISTANCE = 14.0f;  // Maximum distance for interaction (particles won't affect each other beyond this)
const float MAX_SPEED = 2.5f;      // Maximum speed for particles
const float MIN_SPEED = 0.1f;      // Minimum speed for particles
const float MIN_COLLISION_DISTANCE = 10.0f; // Minimum distance for particles to collide and bounce
const float radius = 9.0f;
bool pause = 0;
// Particle struct definition
struct Particle {
    Vector2 position;
    Vector2 velocity;
    Color color;
};

// Function to limit the velocity of a particle to the maximum and minimum speeds
void CapSpeed(Vector2& velocity, float maxSpeed, float minSpeed) {
    float speed = sqrt(velocity.x * velocity.x + velocity.y * velocity.y);
    if (speed > maxSpeed) {
        velocity.x = (velocity.x / speed) * maxSpeed;
        velocity.y = (velocity.y / speed) * maxSpeed;
    }
    if (speed < minSpeed && speed > 0) {
        velocity.x = (velocity.x / speed) * minSpeed;
        velocity.y = (velocity.y / speed) * minSpeed;
    }
}

// Function to handle the interaction between particles (attraction/repulsion)
void HandleInteraction(Particle& p1, Particle& p2) {
    // Calculate the distance between the two particles
    float dx = p2.position.x - p1.position.x;
    float dy = p2.position.y - p1.position.y;
    float distance = sqrt(dx * dx + dy * dy);

    // Only interact if the particles are within a certain distance range
    if (distance < MAX_DISTANCE && distance > MIN_DISTANCE) {
        // Calculate the force (scaled by inverse of distance)
        float force = -FORCE_STRENGTH / distance;

        // Calculate direction of force (normalize vector)
        Vector2 direction = { dx / distance, dy / distance };

        // Apply force (attraction or repulsion)
        Vector2 forceVector = { direction.x * force, direction.y * force };

        // Apply the force to the particles
        p1.velocity.x += forceVector.x;
        p1.velocity.y += forceVector.y;
        p2.velocity.x -= forceVector.x;
        p2.velocity.y -= forceVector.y;
    }

    // Check if the particles are colliding (too close to each other)
    if (distance < MIN_COLLISION_DISTANCE) {
        // Calculate the direction vector for the collision response
        Vector2 collisionDirection = { dx / distance, dy / distance };

        // Apply repulsive force to both particles
        p1.velocity.x -= collisionDirection.x * FORCE_STRENGTH;
        p1.velocity.y -= collisionDirection.y * FORCE_STRENGTH;
        p2.velocity.x += collisionDirection.x * FORCE_STRENGTH;
        p2.velocity.y += collisionDirection.y * FORCE_STRENGTH;
    }
    // Check if the particles are colliding (too close to each other)
    if (distance < MIN_COLLISION_DISTANCE) {
        // Calculate the direction vector for the collision response
        Vector2 collisionDirection = { dx / distance, dy / distance };

        // Apply repulsive force to both particles
        p1.velocity.x -= collisionDirection.x * FORCE_STRENGTH;
        p1.velocity.y -= collisionDirection.y * FORCE_STRENGTH;
        p2.velocity.x += collisionDirection.x * FORCE_STRENGTH;
        p2.velocity.y += collisionDirection.y * FORCE_STRENGTH;
    }
}

void InitializeParticles(vector<Particle>& particles) {
    for (Particle& particle : particles) {
        particle.position = { (float)(rand() % screenWidth), (float)(rand() % screenHeight) };
        particle.velocity = { (float)(rand() % 5 - 2), (float)(rand() % 5 - 2) };
        particle.color = Color{ (unsigned char)(rand() % 256), (unsigned char)(rand() % 256),
                                (unsigned char)(rand() % 256), 255 };
    }
}

void UpdateParticles(vector<Particle>& particles) {

    for (int i = 0; i < MAX_PARTICLES; i++) {

        particles[i].position.x += particles[i].velocity.x;
        particles[i].position.y += particles[i].velocity.y;

        // Cap the speed of the particles (both max and min speed)
        CapSpeed(particles[i].velocity, MAX_SPEED, MIN_SPEED);

        // Bounce off the edges of the screen (left, right, top, bottom)
        if (particles[i].position.x >= screenWidth - radius || particles[i].position.x <= radius) {
            particles[i].velocity.x *= -1.0;
            if (particles[i].position.x >= screenWidth - radius) {
                particles[i].position.x = screenWidth - radius;
            }
            else {
                particles[i].position.x = radius;
            }

        }
        if (particles[i].position.y >= screenHeight - radius || particles[i].position.y <= radius) {
            particles[i].velocity.y *= -1.0;
            if (particles[i].position.y >= screenHeight - radius) {
                particles[i].position.y = screenHeight - radius;
            }
            else {
                particles[i].position.y = radius;
            }
        }

    }
}


void checkKeyBoardInput(vector<Particle>& particles) {

    if (IsKeyPressed(KEY_DOWN)) {
        for (Particle& particle : particles) {
            particle.velocity = { 0.0,MAX_SPEED };
        }
    }
    if (IsKeyPressed(KEY_UP)) {
        for (Particle& particle : particles) {
            particle.velocity = { 0.0,-MAX_SPEED };
        }
    }
    if (IsKeyPressed(KEY_LEFT)) {
        for (Particle& particle : particles) {
            particle.velocity = { -MAX_SPEED,0.0 };
        }
    }
    if (IsKeyPressed(KEY_RIGHT)) {
        for (Particle& particle : particles) {
            particle.velocity = { MAX_SPEED,0.0 };
        }
    }
    if (IsKeyPressed(KEY_SPACE)) {
        pause = !pause;
    }

}

int main() {
    // Set up window
    InitWindow(screenWidth, screenHeight, "Multiple Particle Interaction");

    srand(static_cast<unsigned int>(time(0)));

    // Array of particles
    vector<Particle> particles(MAX_PARTICLES);

    // Initialize particles with random properties
    InitializeParticles(particles);

    SetTargetFPS(144);

    while (!WindowShouldClose()) {

        checkKeyBoardInput(particles);
        if (!pause) {

            UpdateParticles(particles);

            // Particle interaction (attraction/repulsion and collision)
            for (int i = 0; i < MAX_PARTICLES; i++) {
                for (int j = i + 1; j < MAX_PARTICLES; j++) {
                    HandleInteraction(particles[i], particles[j]);
                }
            }
        }

        BeginDrawing();
        ClearBackground(BLACK);

        for (int i = 0; i < MAX_PARTICLES; i++) {
            DrawCircleV(particles[i].position, radius, particles[i].color);
        }

        DrawText(TextFormat("FPS: %i", GetFPS()), 10, 10, 20, WHITE);
        EndDrawing();
    }

    CloseWindow();
    return 0;
}
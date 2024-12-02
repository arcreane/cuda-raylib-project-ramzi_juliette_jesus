#include <raylib.h>
#include <cstdlib>
#include <ctime>
#include <cmath>

// Constants for controlling the number of particles and interaction forces
#define MAX_PARTICLES 500
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
}

int main() {
    // Set up window
    int screenWidth = 1440;
    int screenHeight = 920;
    InitWindow(screenWidth, screenHeight, "Multiple Particle Interaction");

    srand(static_cast<unsigned int>(time(0)));

    // Array of particles
    Particle particles[MAX_PARTICLES];

    // Initialize particles with random properties
    for (int i = 0; i < MAX_PARTICLES; i++) {
        particles[i].position = { (float)(rand() % screenWidth), (float)(rand() % screenHeight) };
        particles[i].velocity = { (float)(rand() % 5 - 2), (float)(rand() % 5 - 2) };
        particles[i].color = Color{ (unsigned char)(rand() % 256), (unsigned char)(rand() % 256),
                                    (unsigned char)(rand() % 256), 255 };
    }

    SetTargetFPS(144);

    while (!WindowShouldClose()) {
        for (int i = 0; i < MAX_PARTICLES; i++) {
            particles[i].position.x += particles[i].velocity.x;
            particles[i].position.y += particles[i].velocity.y;

            // Cap the speed of the particles (both max and min speed)
            CapSpeed(particles[i].velocity, MAX_SPEED, MIN_SPEED);

            // Bounce off the edges of the screen (left, right, top, bottom)
            if (particles[i].position.x >= screenWidth || particles[i].position.x <= 0) {
                particles[i].velocity.x *= -1;
            }
            if (particles[i].position.y >= screenHeight || particles[i].position.y <= 0) {
                particles[i].velocity.y *= -1;
            }
        }

        // Particle interaction (attraction/repulsion and collision)
        for (int i = 0; i < MAX_PARTICLES; i++) {
            for (int j = i + 1; j < MAX_PARTICLES; j++) {
                HandleInteraction(particles[i], particles[j]);
            }
        }

        BeginDrawing();
        ClearBackground(BLACK);

        for (int i = 0; i < MAX_PARTICLES; i++) {
            DrawCircleV(particles[i].position, 7.0f, particles[i].color);
        }

        DrawText(TextFormat("FPS: %i", GetFPS()), 10, 10, 20, WHITE);
        EndDrawing();
    }

    CloseWindow();
    return 0;
}

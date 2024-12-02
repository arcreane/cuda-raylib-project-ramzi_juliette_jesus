#include <raylib.h>
#include <cstdlib>  // For rand() and srand()
#include <ctime>    // For time
#include <cmath>    // For sqrt()

// Particle struct definition
struct Particle {
    Vector2 position;
    Vector2 velocity;
    Color color;
};

// Constants for controlling the number of particles and interaction forces
#define MAX_PARTICLES 500  // Number of particles

// Attraction/repulsion force constant
const float FORCE_STRENGTH = 5.0f;  // You can tweak this to adjust force intensity
const float MIN_DISTANCE = 8.0f;   // Minimum distance for interaction (avoid division by zero)
const float MAX_DISTANCE = 14.0f;  // Maximum distance for interaction (particles won't affect each other beyond this)

// Maximum particle speed
const float MAX_SPEED = 4.0f; // Maximum speed for particles

// Function to limit the velocity of a particle to the maximum speed
void CapSpeed(Vector2& velocity, float maxSpeed) {
    // Calculate the magnitude (length) of the velocity vector
    float speed = sqrt(velocity.x * velocity.x + velocity.y * velocity.y);

    // If the speed exceeds the max speed, normalize and scale the velocity
    if (speed > maxSpeed) {
        velocity.x = (velocity.x / speed) * maxSpeed;
        velocity.y = (velocity.y / speed) * maxSpeed;
    }
}

int main() {
    // Set up window
    int screenWidth = 1440;
    int screenHeight = 920;
    InitWindow(screenWidth, screenHeight, "Multiple Particle Interaction");

    // Seed the random number generator
    srand(static_cast<unsigned int>(time(0)));

    // Array of particles
    Particle particles[MAX_PARTICLES];

    // Initialize particles with random properties
    for (int i = 0; i < MAX_PARTICLES; i++) {
        particles[i].position = { (float)(rand() % screenWidth), (float)(rand() % screenHeight) };  // Random position
        particles[i].velocity = { (float)(rand() % 5 - 2), (float)(rand() % 5 - 2) };  // Random velocity (-2 to 2)
        particles[i].color = Color{ (unsigned char)(rand() % 256), (unsigned char)(rand() % 256),
                                    (unsigned char)(rand() % 256), 255 };  // Random color
    }

    // Set the frame rate
    SetTargetFPS(144);

    // Main game loop
    while (!WindowShouldClose()) {
        // Update particle positions based on their velocities
        for (int i = 0; i < MAX_PARTICLES; i++) {
            particles[i].position.x += particles[i].velocity.x;
            particles[i].position.y += particles[i].velocity.y;

            // Cap the speed of the particles
            CapSpeed(particles[i].velocity, MAX_SPEED);

            // Bounce off the edges of the screen (left, right, top, bottom)
            if (particles[i].position.x >= screenWidth || particles[i].position.x <= 0) {
                particles[i].velocity.x *= -1;  // Reverse horizontal velocity
            }
            if (particles[i].position.y >= screenHeight || particles[i].position.y <= 0) {
                particles[i].velocity.y *= -1;  // Reverse vertical velocity
            }
        }

        // Particle interaction (attraction/repulsion)
        for (int i = 0; i < MAX_PARTICLES; i++) {
            for (int j = i + 1; j < MAX_PARTICLES; j++) {
                // Calculate the distance between particle i and particle j
                float dx = particles[j].position.x - particles[i].position.x;
                float dy = particles[j].position.y - particles[i].position.y;
                float distance = sqrt(dx * dx + dy * dy);

                if (distance < MAX_DISTANCE && distance > MIN_DISTANCE) {
                    // Calculate the force (scaled by inverse of distance)
                    float force = -FORCE_STRENGTH / distance;

                    // Calculate direction of force (normalize vector)
                    Vector2 direction = { dx / distance, dy / distance };

                    // Apply force (attraction/repulsion)
                    Vector2 forceVector = { direction.x * force, direction.y * force };

                    // Apply repulsion force to both particles
                    particles[i].velocity.x += forceVector.x;
                    particles[i].velocity.y += forceVector.y;
                    particles[j].velocity.x -= forceVector.x;  // Opposite direction for the other particle
                    particles[j].velocity.y -= forceVector.y;  // Opposite direction for the other particle
                }
            }
        }

        // Start drawing
        BeginDrawing();
        ClearBackground(BLACK);

        // Draw all particles
        for (int i = 0; i < MAX_PARTICLES; i++) {
            DrawCircleV(particles[i].position, 5.0f, particles[i].color);
        }

        // Display the FPS in the top-left corner
        DrawText(TextFormat("FPS: %i", GetFPS()), 10, 10, 20, WHITE);

        // End drawing
        EndDrawing();
    }

    // Close window
    CloseWindow();

    return 0;
}

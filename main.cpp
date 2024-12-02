#include <raylib.h>
#include <cstdlib>  // For rand() and srand()
#include <ctime>    // For time
#include <cmath>    // For sqrt()

// Particle struct definition
struct Particle {
    Vector2 position;
    Vector2 velocity;
    Color color;
    float radius;  // Radius of the particle, which we will use as mass
};

// Constants for controlling the number of particles and interaction forces
#define MAX_PARTICLES 1000  // Number of particles 

// Attraction/repulsion force constant
const float FORCE_STRENGTH = 5.0f;  // You can tweak this to adjust force intensity
const float MIN_DISTANCE = 8.0f;   // Minimum distance for interaction (avoid division by zero)
const float MAX_DISTANCE = 14.0f;  // Maximum distance for interaction (particles won't affect each other beyond this)

// Maximum particle speed
const float MAX_SPEED = 8.0f; // Maximum speed for particles
const float MIN_SPEED = 0.5f; // Maximum speed for particles

// Function to limit the velocity of a particle to the maximum and minimum speeds
void CapSpeed(Vector2& velocity, float maxSpeed, float minSpeed) {
    // Calculate the magnitude (length) of the velocity vector
    float speed = sqrt(velocity.x * velocity.x + velocity.y * velocity.y);

    // If the speed exceeds the max speed, normalize and scale the velocity
    if (speed > maxSpeed) {
        velocity.x = (velocity.x / speed) * maxSpeed;
        velocity.y = (velocity.y / speed) * maxSpeed;
    }

    // If the speed is below the minimum speed, normalize and scale the velocity
    if (speed < minSpeed && speed > 0) {
        velocity.x = (velocity.x / speed) * minSpeed;
        velocity.y = (velocity.y / speed) * minSpeed;
    }
}
// Function to calculate the mass based on the radius of the particle
float CalculateMass(float radius) {
    return radius * radius * radius;  // Proportional to the volume of a sphere
}

// Function to handle particle collision and momentum conservation
void HandleCollision(Particle& p1, Particle& p2) {
    // Calculate the distance between the particles
    float dx = p2.position.x - p1.position.x;
    float dy = p2.position.y - p1.position.y;
    float distance = sqrt(dx * dx + dy * dy);

    // If the particles are close enough, calculate the collision
    if (distance < p1.radius + p2.radius) {
        // Calculate the mass of each particle (using radius as mass)
        float mass1 = CalculateMass(p1.radius);
        float mass2 = CalculateMass(p2.radius);

        // Calculate the velocity vectors after collision (Conservation of Momentum)
        Vector2 v1 = p1.velocity;
        Vector2 v2 = p2.velocity;

        // For newV1 and newV2, manually scale the x and y components of the vectors.
        Vector2 newV1 = { (v1.x * (mass1 - mass2) + v2.x * 2 * mass2) / (mass1 + mass2),
                          (v1.y * (mass1 - mass2) + v2.y * 2 * mass2) / (mass1 + mass2) };

        Vector2 newV2 = { (v2.x * (mass2 - mass1) + v1.x * 2 * mass1) / (mass1 + mass2),
                          (v2.y * (mass2 - mass1) + v1.y * 2 * mass1) / (mass1 + mass2) };


        // Update velocities to conserve momentum
        p1.velocity = newV1;
        p2.velocity = newV2;
    }
}

int main() {
    // Set up window
    int screenWidth = 1440;
    int screenHeight = 920;
    InitWindow(screenWidth, screenHeight, "Multiple Particle Interaction with Momentum Conservation");

    // Seed the random number generator
    srand(static_cast<unsigned int>(time(0)));

    // Array of particles
    Particle particles[MAX_PARTICLES];

    // Initialize particles with random properties
    for (int i = 0; i < MAX_PARTICLES; i++) {
        particles[i].position = { (float)(rand() % screenWidth), (float)(rand() % screenHeight) };  // Random position
        particles[i].velocity = { (float)(rand() % 5 - 2), (float)(rand() % 5 - 2) };  // Random velocity (-2 to 2)
        particles[i].radius = (float)(rand() % 10 + 5); // Random radius (5 to 15)
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
            CapSpeed(particles[i].velocity, MAX_SPEED,MIN_SPEED);

            // Bounce off the edges of the screen (left, right, top, bottom)
            if (particles[i].position.x >= screenWidth || particles[i].position.x <= 0) {
                particles[i].velocity.x *= -1;  // Reverse horizontal velocity
            }
            if (particles[i].position.y >= screenHeight || particles[i].position.y <= 0) {
                particles[i].velocity.y *= -1;  // Reverse vertical velocity
            }
        }

        // Particle interaction (attraction/repulsion and momentum conservation)
        for (int i = 0; i < MAX_PARTICLES; i++) {
            for (int j = i + 1; j < MAX_PARTICLES; j++) {
                HandleCollision(particles[i], particles[j]);
            }
        }

        // Start drawing
        BeginDrawing();
        ClearBackground(BLACK);

        // Draw all particles
        for (int i = 0; i < MAX_PARTICLES; i++) {
            DrawCircleV(particles[i].position, particles[i].radius, particles[i].color);
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

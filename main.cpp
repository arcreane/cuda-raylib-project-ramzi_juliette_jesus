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

// Constants for Attraction/Repulsion Force
const float MIN_DISTANCE = 10.0f;   // Minimum distance for interaction (to avoid division by zero)
const float MAX_DISTANCE = 20.0f;  // Maximum distance for interaction (particles won't affect each other beyond this)

int main() {
    // Set up window
    int screenWidth = 800;
    int screenHeight = 600;
    InitWindow(screenWidth, screenHeight, "Particle Interaction Test");

    // Seed the random number generator
    srand(static_cast<unsigned int>(time(0)));

    // Initialize two particles
    Particle particles[2];

    // Particle 1: Moving left to right
    particles[0].position = { 100.0f, screenHeight / 2.0f };
    particles[0].velocity = { 3.0f, 0.0f };  // Moving right
    particles[0].color = RED;

    // Particle 2: Moving right to left
    particles[1].position = { screenWidth - 100.0f, screenHeight / 2.0f };
    particles[1].velocity = { -3.0f, 0.0f }; // Moving left
    particles[1].color = BLUE;

    // Set the frame rate
    SetTargetFPS(144);

    // Main game loop
    while (!WindowShouldClose()) {
        // Update particle positions based on their velocities
        for (int i = 0; i < 2; i++) {
            particles[i].position.x += particles[i].velocity.x;
            particles[i].position.y += particles[i].velocity.y;

            // Bounce off the edges of the screen (left, right, top, bottom)
            if (particles[i].position.x >= screenWidth || particles[i].position.x <= 0) {
                particles[i].velocity.x *= -1;  // Reverse horizontal velocity
            }
            if (particles[i].position.y >= screenHeight || particles[i].position.y <= 0) {
                particles[i].velocity.y *= -1;  // Reverse vertical velocity
            }
        }

        // Particle interaction (repulsion between two particles)
        for (int i = 0; i < 2; i++) {
            for (int j = i + 1; j < 2; j++) {
                // Calculate the distance between particle i and particle j
                float dx = particles[j].position.x - particles[i].position.x;
                float dy = particles[j].position.y - particles[i].position.y;
                float distance = sqrt(dx * dx + dy * dy);

                // Only apply repulsion if the particles are within the interaction range
                if (distance < MAX_DISTANCE && distance > MIN_DISTANCE) {

                    // Apply the repulsion force to both particles
                    particles[i].velocity.x *= -1;
                    particles[i].velocity.y *= -1;
                    particles[j].velocity.x *= -1;  // Opposite direction for the other particle
                    particles[j].velocity.y *= -1;  // Opposite direction for the other particle
                }
            }
        }

        // Start drawing
        BeginDrawing();
        ClearBackground(BLACK);

        // Draw all particles
        for (int i = 0; i < 2; i++) {
            DrawCircleV(particles[i].position, 10.0f, particles[i].color);
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

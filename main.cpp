#include <raylib.h>
#include <cstdlib>  // For rand() and srand()
#include <ctime>    // For time()

// Particle struct definition
struct Particle {
    Vector2 position;
    Vector2 velocity;
    Color color;
};

#define MAX_PARTICLES_9_FPS 4e4 // It gives 9 FPS
#define MAX_PARTICLES_35_FPS 1e4 // It gives 35 FPS
int main() {
    // Set up window
    int screenWidth = 1440;
    int screenHeight = 920;
    InitWindow(screenWidth, screenHeight, "Particle System");

    // Seed the random number generator
    srand(static_cast<unsigned int>(time(0)));

    // Number of particles
    const int numParticles = MAX_PARTICLES_35_FPS;

    // Array of particles
    Particle particles[numParticles];

    // Initialize particles with random properties
    for (int i = 0; i < numParticles; i++) {
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
        for (int i = 0; i < numParticles; i++) {
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

        // Start drawing
        BeginDrawing();
        ClearBackground(BLACK);

        // Draw all particles
        for (int i = 0; i < numParticles; i++) {
            DrawCircleV(particles[i].position, 5.0f, particles[i].color);
        }

        // Display the FPS in the top-left corner
        DrawText(TextFormat("FPS: %i", GetFPS()), 10, 10, 50, WHITE);

        // End drawing
        EndDrawing();
    }

    // Close window
    CloseWindow();

    return 0;
}

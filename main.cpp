#include <raylib.h>
#include <cstdlib>  // For rand() and srand()
#include <ctime>    // For time()

// Particle struct definition
struct Particle {
    Vector2 position;
    Vector2 velocity;
    Color color;
};

int main() {
    // Set up window
    int screenWidth = 1280;
    int screenHeight = 600;
    InitWindow(screenWidth, screenHeight, "Particle System");

    // Seed the random number generator
    srand(static_cast<unsigned int>(time(0)));

    // Number of particles
    const int numParticles = 1000;

    // Array of particles
    Particle particles[numParticles];

    // Initialize particles with random properties
    for (int i = 0; i < numParticles; i++) {
        particles[i].position = { (float)(rand() % screenWidth), (float)(rand() % screenHeight) };  // Random position
        particles[i].velocity = { 5 };  // Random velocity (-2 to 2)
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

            // If the particle goes off the screen, reset it to a random position
            if (particles[i].position.x > screenWidth || particles[i].position.x < 0) {
                particles[i].position.x = rand() % screenWidth;
            }
            if (particles[i].position.y > screenHeight || particles[i].position.y < 0) {
                particles[i].position.y = rand() % screenHeight;
            }
        }

        // Start drawing
        BeginDrawing();
        ClearBackground(RAYWHITE);

        // Draw all particles
        for (int i = 0; i < numParticles; i++) {
            DrawCircleV(particles[i].position, 8.0f, particles[i].color);
        }

        // Display the FPS in the top-left corner
        DrawText(TextFormat("FPS: %i", GetFPS()), 10, 10, 20, DARKGRAY);

        // End drawing
        EndDrawing();
    }

    // Close window
    CloseWindow();

    return 0;
}

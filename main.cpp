#include <raylib.h>

int main() {
    // Set the screen width and height
    int screenWidth = 800;
    int screenHeight = 600;

    // Initialize the window
    InitWindow(screenWidth, screenHeight, "My first RAYLIB program!");

    // Set the frame rate to 60 FPS
    SetTargetFPS(60);

    // Main game loop
    while (!WindowShouldClose()) {
        // Start drawing
        BeginDrawing();

        ClearBackground(BLACK);

        DrawText("Welcome to Raylib!", 190, 200, 20, WHITE);

        // End drawing
        EndDrawing();
    }

    // Close the window and OpenGL context
    CloseWindow();

    return 0;
}

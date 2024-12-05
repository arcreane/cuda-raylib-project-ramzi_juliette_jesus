#include <raylib.h>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <vector>

using namespace std;

// Constants for controlling the number of particles and interaction forces
#define MAX_PARTICLES 500
// 2000 = 14 FPS
// 1000 = 53 FPS
// 500 = 142 FPS
const int screenWidth = 1440;
const int screenHeight = 920;
const float radius = 9.0f;          // radius of balls
const float FORCE_STRENGTH = 5.0f;  // Attraction/repulsion force constant
const float MIN_DISTANCE = 2.0f * radius;   // Minimum distance for interaction (avoid division by zero)
const float MAX_DISTANCE = 2.8f * radius;  // Maximum distance for interaction (particles won't affect each other beyond this)
const float MAX_SPEED = 3.5f;      // Maximum speed for particles
const float MIN_SPEED = 0.1f;      // Minimum speed for particles
const float MIN_COLLISION_DISTANCE = 2.5f * radius; // Minimum distance for particles to collide and bounce
const float radius_force = 80.0f;   // Radius of force field created when user click on the screen


float radius_game = 0;
bool pause = 0;
bool mode = 0;
bool start_flag = 0;
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
    //if (distance < MIN_COLLISION_DISTANCE) {
    //    // Calculate the direction vector for the collision response
    //    Vector2 collisionDirection = { dx / distance, dy / distance };

    //    // Apply repulsive force to both particles
    //    p1.velocity.x -= collisionDirection.x * FORCE_STRENGTH;
    //    p1.velocity.y -= collisionDirection.y * FORCE_STRENGTH;
    //    p2.velocity.x += collisionDirection.x * FORCE_STRENGTH;
    //    p2.velocity.y += collisionDirection.y * FORCE_STRENGTH;
    //}

    if (CheckCollisionCircles(Vector2{ p1.position.x,p1.position.y }, radius, Vector2{ p2.position.x,p2.position.y }, radius)) {
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

        // Cap the speed of the particles (both max and min speed)
        CapSpeed(particles[i].velocity, MAX_SPEED, MIN_SPEED);


        particles[i].position.x += particles[i].velocity.x;
        particles[i].position.y += particles[i].velocity.y;



        // Bounce off the edges of the screen (left, right, top, bottom)
        if (particles[i].position.x >= screenWidth - radius || particles[i].position.x <= radius) {
            particles[i].velocity.x *= -1.0;
            if (particles[i].position.x >= screenWidth - radius) {
                particles[i].position.x = (float)screenWidth - radius;
            }
            else {
                particles[i].position.x = radius;
            }

        }
        if (particles[i].position.y >= screenHeight - radius || particles[i].position.y <= radius) {
            particles[i].velocity.y *= -1.0;
            if (particles[i].position.y >= screenHeight - radius) {
                particles[i].position.y = (float)screenHeight - radius;
            }
            else {
                particles[i].position.y = radius;
            }
        }

    }
}


void checkKeyBoardInput(vector<Particle>& particles) {

    if (!mode) {
        if (IsKeyPressed(KEY_P)) {

            for (Particle& particle : particles) {
                particle.velocity = { 0.0,0.0 };
            }

        }
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
        if (IsKeyPressed(KEY_ENTER)) {

            InitializeParticles(particles);
        }
    }

    if (IsKeyPressed(KEY_SPACE)) {
        pause = !pause;
    }

    if (IsKeyPressed(KEY_ENTER)) {

        InitializeParticles(particles);

    }

    if (IsKeyPressed(KEY_M)) {
        mode = !mode;
        start_flag = 0;
        InitializeParticles(particles);

    }

}

void ForceField(vector<Particle>& particles, Vector2& MousePosition) {

    for (int i = 0; i < MAX_PARTICLES; i++) {

        if (particles[i].position.x >= MousePosition.x - radius_force && particles[i].position.x <= MousePosition.x + radius_force
            && particles[i].position.y >= MousePosition.y - radius_force && particles[i].position.y <= MousePosition.y + radius_force) {

            float dx = particles[i].position.x - MousePosition.x;
            float dy = particles[i].position.y - MousePosition.y;
            float distance = sqrt(dx * dx + dy * dy);
            Vector2 Direction = { dx / distance, dy / distance };
            particles[i].velocity.x = Direction.x * MAX_SPEED;
            particles[i].velocity.y = Direction.y * MAX_SPEED;

        }
    }
}


int main() {
    // Set up window
    Vector2 MousePosition;
    InitWindow(screenWidth, screenHeight, "Multiple Particle Interaction");

    // Array of particles
    vector<Particle> particles(MAX_PARTICLES);

    // Initialize particles with random properties
    InitializeParticles(particles);

    SetTargetFPS(144);

    while (!WindowShouldClose()) {


        MousePosition = GetMousePosition();

        checkKeyBoardInput(particles);

        if (!mode) {


            if (!pause) {

                if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
                    ForceField(particles, MousePosition);
                }

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

        }

        else {


            BeginDrawing();
            ClearBackground(BLACK);

            if (!start_flag) {

                // Define the box dimensions
                int boxWidth = 200;
                int boxHeight = 100;
                float boxX = (screenWidth - boxWidth) / 2;
                float boxY = (screenHeight - boxHeight) / 2;

                Rectangle rect1 = { boxX, boxY - 120, boxWidth, boxHeight };
                Rectangle rect2 = { boxX, boxY, boxWidth, boxHeight };
                Rectangle rect3 = { boxX, boxY + 120, boxWidth, boxHeight };
                // Draw the box
                DrawRectangle(rect1.x, rect1.y, rect1.width, rect1.height, BLUE);
                DrawRectangle(rect2.x, rect2.y, rect2.width, rect2.height, VIOLET);
                DrawRectangle(rect3.x, rect3.y, rect3.width, rect3.height, ORANGE);
                DrawRectangleLines(rect1.x, rect1.y, rect1.width, rect1.height, GRAY);
                DrawRectangleLines(rect2.x, rect2.y, rect2.width, rect2.height, GRAY);
                DrawRectangleLines(rect3.x, rect3.y, rect3.width, rect3.height, GRAY);

                // Draw the message to choose the level
                const char* message = "CHOOSE YOUR LEVEL";
                int fontSize = 20;
                int textWidth = MeasureText(message, fontSize);
                DrawText(message, (screenWidth - textWidth) / 2, boxY - 200, fontSize, WHITE);

                // Draw the message "EASY" centered in the first box
                const char* message1 = "EASY";
                int textWidth1 = MeasureText(message1, fontSize);
                int textX = rect1.x + (rect1.width - textWidth1) / 2;
                int textY = rect1.y + (rect1.height - fontSize) / 2;
                DrawText(message1, textX, textY, fontSize, WHITE);

                // Draw the message "MEDIUM" centered in the second box
                const char* message2 = "MEDIUM";
                int textWidth2 = MeasureText(message2, fontSize);
                int textX2 = rect2.x + (rect2.width - textWidth2) / 2;
                int textY2 = rect2.y + (rect2.height - fontSize) / 2;
                DrawText(message2, textX2, textY2, fontSize, WHITE);

                // Draw the message "MEDIUM" centered in the second box
                const char* message3 = "LEGENDARY";
                int textWidth3 = MeasureText(message3, fontSize);
                int textX3 = rect3.x + (rect3.width - textWidth3) / 2;
                int textY3 = rect3.y + (rect3.height - fontSize) / 2;
                DrawText(message3, textX3, textY3, fontSize, WHITE);

                if (CheckCollisionPointRec(MousePosition, rect1)) {
                    if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
                        start_flag = true;
                        radius_game = 80.0f;
                    }
                }
                if (CheckCollisionPointRec(MousePosition, rect2)) {
                    if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
                        start_flag = true;
                        radius_game = 50.0f;
                    }
                }
                if (CheckCollisionPointRec(MousePosition, rect3)) {
                    if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
                        start_flag = true;
                        radius_game = 30.0f;
                    }
                }
            }
            else {

                // Define the box dimensions
                int boxWidth = 200;
                int boxHeight = 100;
                int boxX = (screenWidth - boxWidth) / 2;
                int boxY = (screenHeight - boxHeight) / 2;

                // Draw the message "Terminé" centered in the box
                const char* message = "!!!! CONGRATS !!!!";
                int fontSize = 50;
                int textWidth = MeasureText(message, fontSize);
                int textX = boxX + (boxWidth - textWidth) / 2;
                int textY = boxY + (boxHeight - fontSize) / 2;

                DrawText(message, textX, textY, fontSize, GREEN);

            }

        }
        DrawText(TextFormat("FPS: %i", GetFPS()), 10, 10, 20, WHITE);
        EndDrawing();
    }


    CloseWindow();
    return 0;

}


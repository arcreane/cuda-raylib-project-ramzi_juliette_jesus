﻿
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <raylib.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include<iostream>
int main()
{

    Color Dark_Green = Color{ 20, 160, 133, 255 };

    const int screenWidth = 800;
    const int screenHeight = 600;  
    int ball_x = 100;
    int ball_y = 100;
    int ball_speed_x = 5;
    int ball_speed_y = 5;
    int ball_radius = 15;

    // hello

    std::cout << "Hello World" << std::endl;

    InitWindow(screenWidth, screenHeight, "My first RAYLIB program!");
    SetTargetFPS(60);

    while (WindowShouldClose() == false) {
        BeginDrawing();
        ClearBackground(Dark_Green);
        ball_x += ball_speed_x;
        ball_y += ball_speed_y;

        if (ball_x + ball_radius >= screenWidth || ball_x - ball_radius <= 0)
        {
            ball_speed_x *= -1;
        }

        if (ball_y + ball_radius >= screenHeight || ball_y - ball_radius <= 0)
        {
            ball_speed_y *= -1;
        }

        DrawCircle(ball_x, ball_y, ball_radius, WHITE);
        EndDrawing();
    }

    CloseWindow();
    return 0;

    return 0;
}

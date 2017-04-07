g++ -Wall main.cpp -lmingw32 -lopengl32 -lglew32 -lSDL2main -lSDL2 -o planet

@if ERRORLEVEL 1 goto end

planet

:end

---
name: coding-principles
description: before and after you write code — ensure it follows the project's coding principles
---
Coding principles.
Code is the way to communicate with the machine, it is critical that it be clear to a person who enters from the outside.
The architecture, structure and names of the objects in the code must tell in the clearest way what is happening in the code. It must be that way.
Updating code - it is sensitive, it must be done gently, not to refactor what you were not asked to do.
3 \ 4 lines must be added at the beginning of the file that explain how this file fits into all the existing code.
Building code requires checking that it actually connects and works.
Code should be elegant, cleverness, tricks, sophisticated things should not be in the code, they should only be in certain cases.
The way of working is - building something, once it is assembled - you make sure that it actually works and continue.
This is a research and development project, it is critical that it be clean. There is no need to maintain backward compatibility at all.
We will always strive for there to be only one clear and direct way to use the code. We will only deviate from this when necessary.
use uv
create clear and modular code.
the functions that do somthing importent are tools, we need to use them in also in JupyterNotebook, in repl.
the code need to be intuative and simple. the code itself is the stroy what the program do.

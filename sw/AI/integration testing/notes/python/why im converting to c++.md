pyuno/demo/hello_world_comp.pyuno
this gives a good example of how inputs and outputs can be implemented

there is one large issue.
the libreoffice code doesnt work as well with python compared ot c++
the code doesnt look throughout the whole code base fo rpython code and adjusting that to make it work fro my code will be bad practice
may have compile time errors as well as the installer may become broken

Speed improvement
there is also the fact that alot of the methods used to make the server and script faster will be overshadowed by python being slower for libre offices use case

    the majority of the libreoffice code is in c++ and it would be beneficial for the open source project for the code to be in c++.

    c++ will be much faster as ti will be compiled directly into machine code.
    it will also give me better control including more control of memory which can help me reach better vram and ram utilisation. It has been a significant issue in my testing so far as it causes battery drain of my laptop due to the power needed to run the ai. however speed is fine in mmy pc.
    Better concurrency support is available in C++ with more libraries i can use to make the code run faster and more efficiently on more systems
    Imporved start up speed

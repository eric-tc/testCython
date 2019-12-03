cmake_minimum_required(VERSION 2.8.9)
project (EmbedPython)
 
#For the shared library:
set(PYTHON_INCLUDE_PATH "$ENV{HOME}/anaconda3/include/python3.7m")
set(PYTHON_LIBRARIES "$ENV{HOME}/anaconda3/lib/libpython3.7m.so")

set (PROJECT_LINK_LIBS "/home/velab/VsCode++/TestEmbedPython/lib/libtrain.cpython-37m-x86_64-linux-gnu.so" )

link_directories(${PROJECT_LINK_LIBS})


#Libreria dinamica python
link_directories(${PYTHON_LIBRARIES}  )
 
 #include delle librerie
include_directories(${PYTHON_INCLUDE_PATH})
 
add_executable(EmbedPython main.cpp)
target_link_libraries(EmbedPython ${PYTHON_LIBRARIES} ${PROJECT_LINK_LIBS} )